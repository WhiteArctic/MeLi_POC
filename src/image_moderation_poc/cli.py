from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from image_moderation_poc import config
from image_moderation_poc.datasets import build_dataset_splits, stable_hash
from image_moderation_poc.detector import ImageModerationService
from image_moderation_poc.evaluation import evaluate_predictions, save_evaluation_report
from image_moderation_poc.text_model import TextNaiveBayes


def build_datasets(_: argparse.Namespace) -> None:
    paths = build_dataset_splits()
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def train_model(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.train_csv)
    if args.max_rows_per_class:
        sampled = []
        for _, class_frame in frame.groupby("target_has_infraction"):
            sampled.append(
                class_frame.assign(
                    sample_hash=class_frame["leakage_group_id"].apply(lambda v: stable_hash(str(v)))
                )
                .sort_values("sample_hash")
                .head(args.max_rows_per_class)
                .drop(columns=["sample_hash"])
            )
        frame = pd.concat(sampled, ignore_index=True)
    model = TextNaiveBayes.train(
        frame["ocr_text"].fillna("").astype(str).tolist(),
        frame["target_has_infraction"].astype(bool).tolist(),
        feature_count=args.feature_count,
        alpha=args.alpha,
    )
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}")


def evaluate_model(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.dataset_csv).reset_index(drop=True)
    if args.max_rows:
        frame = frame.head(args.max_rows).copy()
    model = TextNaiveBayes.load(args.model_path) if args.model_path.exists() else None
    service = ImageModerationService(model=model)

    predictions = []
    texts = frame["ocr_text"].fillna("").astype(str).tolist()
    if args.mode == "model_only":
        if model is None:
            raise FileNotFoundError(f"Model not found: {args.model_path}")
        predictions = [
            model.predict_proba_one(text) >= args.threshold
            for text in texts
        ]
    else:
        for text in texts:
            predictions.append(service.diagnose_text(text).has_infraction)

    report = evaluate_predictions(frame, predictions)
    report["dataset_csv"] = str(args.dataset_csv)
    report["mode"] = args.mode
    report["threshold"] = args.threshold
    save_evaluation_report(report, args.output_json)
    print(json.dumps(report["overall"], indent=2))


def diagnose_url(args: argparse.Namespace) -> None:
    if args.candidate:
        service = ImageModerationService.from_candidate(args.model_path, verify_ssl=False)
    else:
        service = ImageModerationService.from_model_path(args.model_path, verify_ssl=False)
    diagnosis = service.diagnose_image(args.picture_url)
    print(json.dumps(diagnosis.as_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Image moderation POC CLI")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("build-datasets")
    p.set_defaults(func=build_datasets)

    p = sub.add_parser("train")
    p.add_argument("--train-csv", type=Path, default=config.DATASET_DIR / "train.csv")
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--feature-count", type=int, default=2**18)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-rows-per-class", type=int, default=60000)
    p.set_defaults(func=train_model)

    p = sub.add_parser("evaluate")
    p.add_argument("--dataset-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--output-json", type=Path, default=config.ROOT_DIR / "outputs" / "reports" / "evaluation.json")
    p.add_argument("--mode", choices=["model_only", "hybrid"], default="model_only")
    p.add_argument("--threshold", type=float, default=config.DEFAULT_MODEL_THRESHOLD)
    p.add_argument("--max-rows", type=int, default=0)
    p.set_defaults(func=evaluate_model)

    p = sub.add_parser("diagnose-url")
    p.add_argument("picture_url")
    p.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    p.add_argument("--candidate", action="store_true")
    p.set_defaults(func=diagnose_url)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
