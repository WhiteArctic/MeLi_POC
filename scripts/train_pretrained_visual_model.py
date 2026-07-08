from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc import config
from image_moderation_poc.datasets import stable_hash
from image_moderation_poc.evaluation import compute_metrics, save_evaluation_report
from scripts.train_visual_model import image_cache_path, load_or_download_image


def sample_balanced(frame: pd.DataFrame, rows_per_class: int) -> pd.DataFrame:
    dedup = (
        frame.assign(sample_hash=frame["leakage_group_id"].apply(lambda v: stable_hash(str(v))))
        .sort_values(["sample_hash", "picture_url"])
        .drop_duplicates("leakage_group_id", keep="first")
    )
    sampled = []
    for _, class_frame in dedup.groupby("target_has_infraction", dropna=False):
        sampled.append(class_frame.sort_values("sample_hash").head(rows_per_class))
    return pd.concat(sampled, ignore_index=True).drop(columns=["sample_hash"]).reset_index(drop=True)


class ImageFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = load_or_download_image(str(row["picture_url"]), self.cache_dir)
        return self.transform(image), float(bool(row["target_has_infraction"]))


def extract_embeddings(
    frame: pd.DataFrame,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    dataset = ImageFrameDataset(frame, cache_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = mobilenet_v3_small(weights=weights)
    backbone = nn.Sequential(model.features, model.avgpool, nn.Flatten()).to(device).eval()

    embeddings = []
    labels = []
    seen = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_embeddings = backbone(images).cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.numpy())
            seen += len(batch_labels)
            if progress_every > 0 and seen % progress_every == 0:
                print(f"Embedded {seen}/{len(frame)} images", flush=True)

    return np.vstack(embeddings).astype(np.float32), np.concatenate(labels).astype(np.float32)


def train_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (train_x - mean) / scale
    y = train_y
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    positive_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))

    for _ in range(epochs):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        sample_weights = np.where(y > 0.5, positive_weight, 1.0).astype(np.float32)
        error = (probs - y) * sample_weights
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias, mean, scale


def predict(embeddings: np.ndarray, weights: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray):
    x = (embeddings - mean) / scale
    logits = x @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, min_precision: float):
    best_threshold = 0.99
    best_metrics = None
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = probabilities >= threshold
        metrics = compute_metrics(y_true.astype(bool).tolist(), predictions.tolist()).as_dict()
        if metrics["precision"] >= min_precision and (
            best_metrics is None or metrics["recall"] > best_metrics["recall"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        predictions = probabilities >= best_threshold
        best_metrics = compute_metrics(y_true.astype(bool).tolist(), predictions.tolist()).as_dict()
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MobileNet embedding classifier.")
    parser.add_argument("--train-csv", type=Path, default=config.DATASET_DIR / "train.csv")
    parser.add_argument("--validation-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    parser.add_argument("--rows-per-class", type=int, default=500)
    parser.add_argument("--validation-rows-per-class", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=config.IMAGE_CACHE_DIR)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "reports" / "pretrained_visual_model_validation.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.MODEL_DIR / "pretrained_visual_linear_v1.npz",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--insecure-model-download", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    if args.insecure_model_download:
        ssl._create_default_https_context = ssl._create_unverified_context

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = sample_balanced(pd.read_csv(args.train_csv), args.rows_per_class)
    validation = sample_balanced(pd.read_csv(args.validation_csv), args.validation_rows_per_class)

    train_x, train_y = extract_embeddings(
        train, args.cache_dir, args.batch_size, device, args.progress_every
    )
    validation_x, validation_y = extract_embeddings(
        validation, args.cache_dir, args.batch_size, device, args.progress_every
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
    )
    detail = validation[
        ["picture_url", "target_has_infraction", "segment", "site", "leakage_group_id"]
    ].copy()
    detail["visual_probability"] = probabilities
    detail["prediction"] = probabilities >= threshold
    detail_path = args.output_json.with_suffix(".details.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)

    report = {
        "model_path": str(args.model_path),
        "device": str(device),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "threshold": threshold,
        "metrics": metrics,
        "details_csv": str(detail_path),
    }
    save_evaluation_report(report, args.output_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
