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
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc import config
from image_moderation_poc.datasets import stable_hash
from image_moderation_poc.evaluation import compute_metrics, save_evaluation_report
from scripts.train_visual_model import load_or_download_image


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


class ModerationImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = load_or_download_image(str(row["picture_url"]), self.cache_dir)
        return self.transform(image), torch.tensor(float(bool(row["target_has_infraction"])))


def build_model(trainable_tail_blocks: int, device: torch.device) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if trainable_tail_blocks > 0:
        for block in model.features[-trainable_tail_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    return model.to(device)


def make_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.82, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.15, 0.18, 0.18, 0.04)], p=0.45),
                transforms.RandomRotation(4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return MobileNet_V3_Small_Weights.DEFAULT.transforms()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    positive_weight: float,
) -> nn.Module:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_state = None
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        probabilities, y_true = predict_probabilities(model, validation_loader, device)
        threshold, metrics = choose_threshold(y_true, probabilities, min_precision=0.95)
        score = metrics["recall"] if metrics["precision"] >= 0.95 else 0.0
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"threshold={threshold:.2f} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    y_true = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).squeeze(1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(y_true).astype(bool)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, min_precision: float):
    best_threshold = 0.99
    best_metrics = None
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = probabilities >= threshold
        metrics = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
        if metrics["precision"] >= min_precision and (
            best_metrics is None or metrics["recall"] > best_metrics["recall"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        predictions = probabilities >= best_threshold
        best_metrics = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
    return best_threshold, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3 for image moderation.")
    parser.add_argument("--train-csv", type=Path, default=config.DATASET_DIR / "train.csv")
    parser.add_argument("--validation-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    parser.add_argument("--rows-per-class", type=int, default=800)
    parser.add_argument("--validation-rows-per-class", type=int, default=250)
    parser.add_argument("--cache-dir", type=Path, default=config.IMAGE_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=config.MODEL_DIR / "finetuned_mobilenet_v1.pt")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "reports" / "finetuned_visual_validation.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--trainable-tail-blocks", type=int, default=3)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--insecure-model-download", action="store_true")
    args = parser.parse_args()

    if args.insecure_model_download:
        ssl._create_default_https_context = ssl._create_unverified_context

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = sample_balanced(pd.read_csv(args.train_csv), args.rows_per_class)
    validation = sample_balanced(pd.read_csv(args.validation_csv), args.validation_rows_per_class)

    train_loader = DataLoader(
        ModerationImageDataset(train, args.cache_dir, make_transforms(train=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        ModerationImageDataset(validation, args.cache_dir, make_transforms(train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    positive_weight = float((len(train) - train["target_has_infraction"].sum()) / max(train["target_has_infraction"].sum(), 1))

    model = build_model(args.trainable_tail_blocks, device)
    model = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        positive_weight=positive_weight,
    )

    probabilities, y_true = predict_probabilities(model, validation_loader, device)
    threshold, metrics = choose_threshold(y_true, probabilities, args.min_precision)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "threshold": threshold,
            "model": "mobilenet_v3_small",
            "trainable_tail_blocks": args.trainable_tail_blocks,
        },
        args.model_path,
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
