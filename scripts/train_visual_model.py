from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc import config
from image_moderation_poc.datasets import stable_hash
from image_moderation_poc.evaluation import compute_metrics, save_evaluation_report
from image_moderation_poc.image_io import load_or_download_image as load_or_download_cached_image
from image_moderation_poc.visual_model import VisualFeatureClassifier, extract_visual_features


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


def load_or_download_image(url: str, cache_dir: Path) -> Image.Image:
    return load_or_download_cached_image(
        url,
        cache_dir,
        timeout_seconds=6.0,
        verify_ssl=False,
    ).image


def build_feature_matrix(
    frame: pd.DataFrame,
    cache_dir: Path,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = []
    labels = []
    kept_rows = []
    failures = 0
    for index, row in frame.iterrows():
        try:
            image = load_or_download_image(str(row["picture_url"]), cache_dir)
            features.append(extract_visual_features(image))
            labels.append(bool(row["target_has_infraction"]))
            kept_rows.append(row)
        except Exception as exc:
            failures += 1
            print(f"Image failure at row={index}: {type(exc).__name__}: {exc}", flush=True)

        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"Processed {index + 1}/{len(frame)} images; failures={failures}", flush=True)

    return (
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=bool),
        pd.DataFrame(kept_rows).reset_index(drop=True),
    )


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_precision: float,
) -> tuple[float, dict[str, object]]:
    best_threshold = 0.99
    best_report: dict[str, object] = {}
    for threshold in np.linspace(0.05, 0.99, 95):
        predictions = probabilities >= threshold
        metrics = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
        if metrics["precision"] >= min_precision and metrics["recall"] >= best_report.get("recall", -1):
            best_threshold = float(threshold)
            best_report = metrics
    if not best_report:
        threshold = 0.99
        predictions = probabilities >= threshold
        best_report = compute_metrics(y_true.tolist(), predictions.tolist()).as_dict()
    return best_threshold, best_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight visual classifier.")
    parser.add_argument("--train-csv", type=Path, default=config.DATASET_DIR / "train.csv")
    parser.add_argument("--validation-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    parser.add_argument("--rows-per-class", type=int, default=600)
    parser.add_argument("--validation-rows-per-class", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=config.IMAGE_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=config.VISUAL_MODEL_PATH)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "reports" / "visual_model_validation.json",
    )
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    train = sample_balanced(pd.read_csv(args.train_csv), args.rows_per_class)
    validation = sample_balanced(pd.read_csv(args.validation_csv), args.validation_rows_per_class)

    train_x, train_y, train_rows = build_feature_matrix(
        train,
        args.cache_dir,
        args.progress_every,
    )
    validation_x, validation_y, validation_rows = build_feature_matrix(
        validation,
        args.cache_dir,
        args.progress_every,
    )

    model = VisualFeatureClassifier.train(train_x, train_y, threshold=0.90)
    validation_probabilities = model.predict_proba_features(validation_x)
    threshold, metrics = evaluate_thresholds(
        validation_y,
        validation_probabilities,
        min_precision=args.min_precision,
    )
    model.threshold = threshold
    model.save(args.model_path)

    detail = validation_rows[
        ["picture_url", "target_has_infraction", "segment", "site", "leakage_group_id"]
    ].copy()
    detail["visual_probability"] = validation_probabilities
    detail["prediction"] = validation_probabilities >= threshold
    detail_path = args.output_json.with_suffix(".details.csv")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path, index=False)

    report = {
        "model_path": str(args.model_path),
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "threshold": threshold,
        "metrics": metrics,
        "details_csv": str(detail_path),
    }
    save_evaluation_report(report, args.output_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
