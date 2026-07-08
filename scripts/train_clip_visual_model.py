from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc import config
from image_moderation_poc.evaluation import compute_metrics, save_evaluation_report
from scripts.train_pretrained_visual_model import choose_threshold, predict, sample_balanced, train_linear
from scripts.train_visual_model import load_or_download_image


class ClipImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, preprocess):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = load_or_download_image(str(row["picture_url"]), self.cache_dir)
        label = bool(row["target_has_infraction"])
        return self.preprocess(image), float(label)


def configure_hf_download(insecure: bool, disable_xet: bool) -> None:
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if not insecure:
        return

    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import httpx
        from huggingface_hub import set_client_factory

        set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True, timeout=120))
    except Exception as exc:
        print(f"Could not install insecure HF client: {type(exc).__name__}: {exc}", flush=True)


def load_open_clip_model(model_name: str, pretrained: str, cache_dir: Path, device: torch.device):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_torch is required. Install with: python -m pip install '.[clip]'"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=str(cache_dir),
    )
    model = model.to(device).eval()
    return model, preprocess


def extract_clip_embeddings(
    frame: pd.DataFrame,
    image_cache_dir: Path,
    model_cache_dir: Path,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, preprocess = load_open_clip_model(model_name, pretrained, model_cache_dir, device)
    dataset = ClipImageDataset(frame, image_cache_dir, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = []
    labels = []
    seen = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_embeddings = model.encode_image(images)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embeddings.append(batch_embeddings.cpu().float().numpy())
            labels.append(batch_labels.numpy())
            seen += len(batch_labels)
            if progress_every > 0 and seen % progress_every == 0:
                print(f"Embedded {seen}/{len(frame)} images", flush=True)

    return np.vstack(embeddings).astype(np.float32), np.concatenate(labels).astype(np.float32)


def score_frame(
    frame: pd.DataFrame,
    image_cache_dir: Path,
    model_cache_dir: Path,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: torch.device,
    progress_every: int,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    embeddings, _ = extract_clip_embeddings(
        frame,
        image_cache_dir,
        model_cache_dir,
        model_name,
        pretrained,
        batch_size,
        device,
        progress_every,
    )
    return predict(embeddings, weights, bias, mean, scale)


def evaluate_fusion(
    details_csv: Path,
    visual_probabilities: np.ndarray,
    min_precision: float,
) -> tuple[float, dict[str, object]]:
    details = pd.read_csv(details_csv)
    y_true = details["target_has_infraction"].astype(bool).to_numpy()
    base_predictions = details["prediction"].astype(bool).to_numpy()

    best_threshold = 0.99
    best_metrics: dict[str, object] | None = None
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = base_predictions | (visual_probabilities >= threshold)
        metrics = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
        if metrics["precision"] >= min_precision and (
            best_metrics is None or metrics["recall"] > best_metrics["recall"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        predictions = base_predictions | (visual_probabilities >= best_threshold)
        best_metrics = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear classifier on OpenCLIP image embeddings.")
    parser.add_argument("--train-csv", type=Path, default=config.DATASET_DIR / "train.csv")
    parser.add_argument("--validation-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    parser.add_argument("--rows-per-class", type=int, default=500)
    parser.add_argument("--validation-rows-per-class", type=int, default=200)
    parser.add_argument("--image-cache-dir", type=Path, default=config.IMAGE_CACHE_DIR)
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "model_cache" / "open_clip",
    )
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_DIR / "clip_visual_linear_v1.npz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "reports" / "clip_visual_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--fusion-details-csv", type=Path)
    parser.add_argument("--insecure-hf-download", action="store_true")
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    configure_hf_download(args.insecure_hf_download, args.disable_hf_xet)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train = sample_balanced(pd.read_csv(args.train_csv), args.rows_per_class)
    validation = sample_balanced(pd.read_csv(args.validation_csv), args.validation_rows_per_class)

    train_x, train_y = extract_clip_embeddings(
        train,
        args.image_cache_dir,
        args.model_cache_dir,
        args.model_name,
        args.pretrained,
        args.batch_size,
        device,
        args.progress_every,
    )
    validation_x, validation_y = extract_clip_embeddings(
        validation,
        args.image_cache_dir,
        args.model_cache_dir,
        args.model_name,
        args.pretrained,
        args.batch_size,
        device,
        args.progress_every,
    )

    weights, bias, mean, scale = train_linear(
        train_x,
        train_y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    probabilities = predict(validation_x, weights, bias, mean, scale)
    threshold, metrics = choose_threshold(validation_y, probabilities, args.min_precision)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.model_path,
        weights=weights,
        bias=np.asarray([bias], dtype=np.float32),
        mean=mean,
        scale=scale,
        threshold=np.asarray([threshold], dtype=np.float32),
        model_name=np.asarray([args.model_name]),
        pretrained=np.asarray([args.pretrained]),
    )

    detail = validation[
        ["picture_url", "target_has_infraction", "segment", "site", "leakage_group_id"]
    ].copy()
    detail["visual_probability"] = probabilities
    detail["prediction"] = probabilities >= threshold
    detail_path = args.output_json.with_suffix(".details.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)

    report: dict[str, object] = {
        "model_path": str(args.model_path),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "threshold": threshold,
        "metrics": metrics,
        "details_csv": str(detail_path),
    }

    if args.fusion_details_csv:
        fusion_frame = pd.read_csv(args.fusion_details_csv)
        fusion_probabilities = score_frame(
            fusion_frame,
            args.image_cache_dir,
            args.model_cache_dir,
            args.model_name,
            args.pretrained,
            args.batch_size,
            device,
            args.progress_every,
            weights,
            bias,
            mean,
            scale,
        )
        fusion_threshold, fusion_metrics = evaluate_fusion(
            args.fusion_details_csv,
            fusion_probabilities,
            args.min_precision,
        )
        fusion_detail = fusion_frame.copy()
        fusion_detail["clip_visual_probability"] = fusion_probabilities
        fusion_detail["clip_fusion_prediction"] = (
            fusion_detail["prediction"].astype(bool).to_numpy()
            | (fusion_probabilities >= fusion_threshold)
        )
        fusion_detail_path = args.output_json.with_suffix(".fusion.details.csv")
        fusion_detail.to_csv(fusion_detail_path, index=False)
        report["fusion"] = {
            "details_csv": str(args.fusion_details_csv),
            "threshold": fusion_threshold,
            "metrics": fusion_metrics,
            "fusion_details_csv": str(fusion_detail_path),
        }

    save_evaluation_report(report, args.output_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
