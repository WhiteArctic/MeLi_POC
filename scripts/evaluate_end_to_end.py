from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc import config
from image_moderation_poc.datasets import stable_hash
from image_moderation_poc.detector import ImageModerationService
from image_moderation_poc.evaluation import evaluate_predictions
from image_moderation_poc.ocr import EasyOCRBackend, EnsembleOCR, TesseractCliOCR
from image_moderation_poc.visual_model import MobileNetV3VisualClassifier


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def sample_frame(
    frame: pd.DataFrame,
    max_rows: int,
    rows_per_class: int,
    dedupe_group_key: str,
) -> pd.DataFrame:
    if dedupe_group_key:
        if dedupe_group_key not in frame.columns:
            raise ValueError(f"Missing dedupe group key: {dedupe_group_key}")
        frame = (
            frame.assign(sample_hash=frame[dedupe_group_key].apply(lambda v: stable_hash(str(v))))
            .sort_values(["sample_hash", "picture_url"])
            .drop_duplicates(dedupe_group_key, keep="first")
            .drop(columns=["sample_hash"])
            .reset_index(drop=True)
        )

    if rows_per_class > 0:
        sampled = []
        for _, class_frame in frame.groupby("target_has_infraction", dropna=False):
            sampled.append(
                class_frame.assign(
                    sample_hash=class_frame["leakage_group_id"].apply(lambda v: stable_hash(str(v)))
                )
                .sort_values("sample_hash")
                .head(rows_per_class)
                .drop(columns=["sample_hash"])
            )
        return pd.concat(sampled, ignore_index=True).reset_index(drop=True)

    if max_rows > 0:
        return (
            frame.assign(sample_hash=frame["leakage_group_id"].apply(lambda v: stable_hash(str(v))))
            .sort_values("sample_hash")
            .head(max_rows)
            .drop(columns=["sample_hash"])
            .reset_index(drop=True)
        )

    return frame.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the productive picture_url path.")
    parser.add_argument("--dataset-csv", type=Path, default=config.DATASET_DIR / "validation.csv")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=config.ROOT_DIR / "outputs" / "reports" / "end_to_end_validation.json",
    )
    parser.add_argument("--details-csv", type=Path, default=None)
    parser.add_argument("--preset", choices=["custom", "candidate"], default="custom")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--visual-model-path", type=Path, default=config.PRETRAINED_VISUAL_MODEL_PATH)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--rows-per-class", type=int, default=25)
    parser.add_argument("--dedupe-group-key", default="leakage_group_id")
    parser.add_argument("--ocr-backend", choices=["tesseract", "easyocr", "ensemble"], default="tesseract")
    parser.add_argument("--visual-classifier", choices=["none", "mobilenet"], default="none")
    parser.add_argument("--visual-model-threshold", type=float, default=config.CANDIDATE_VISUAL_FUSION_THRESHOLD)
    parser.add_argument("--ocr-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--ocr-pass-timeout-seconds", type=float, default=1.25)
    parser.add_argument("--no-ocr-warmup", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset_csv)
    frame = sample_frame(frame, args.max_rows, args.rows_per_class, args.dedupe_group_key)

    if args.preset == "candidate":
        args.ocr_backend = "easyocr"
        args.visual_classifier = "mobilenet"

    tesseract = TesseractCliOCR(
        timeout_seconds=args.ocr_timeout_seconds,
        pass_timeout_seconds=args.ocr_pass_timeout_seconds,
    )
    if args.ocr_backend == "easyocr":
        ocr_backend = EasyOCRBackend()
    elif args.ocr_backend == "ensemble":
        ocr_backend = EnsembleOCR((tesseract, EasyOCRBackend()))
    else:
        ocr_backend = tesseract

    visual_classifier = None
    if args.visual_classifier == "mobilenet":
        if not args.visual_model_path.exists():
            raise FileNotFoundError(f"Visual model not found: {args.visual_model_path}")
        visual_classifier = MobileNetV3VisualClassifier.load(
            args.visual_model_path,
            threshold=args.visual_model_threshold,
        )

    service = ImageModerationService.from_model_path(
        args.model_path,
        ocr_backend=ocr_backend,
        visual_classifier=visual_classifier,
        verify_ssl=False,
    )
    service.visual_model_threshold = args.visual_model_threshold

    if not args.no_ocr_warmup:
        ocr_backend.extract_text(Image.new("RGB", (96, 32), "white"))

    predictions: list[bool] = []
    details: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    failed = 0

    for index, row in frame.iterrows():
        started = time.perf_counter()
        error = ""
        try:
            diagnosis = service.diagnose_image(str(row["picture_url"]))
            prediction = bool(diagnosis.has_infraction)
            evidence = diagnosis.evidence
            score = diagnosis.score
            rule_score = diagnosis.rule_score
            model_score = diagnosis.model_score
            semantic_score = diagnosis.semantic_score
            visual_score = diagnosis.visual_score
            visual_model_score = diagnosis.visual_model_score
            ocr_text = diagnosis.ocr_text
        except Exception as exc:
            failed += 1
            prediction = False
            evidence = ""
            score = 0.0
            rule_score = 0.0
            model_score = None
            semantic_score = None
            visual_score = 0.0
            visual_model_score = None
            ocr_text = ""
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000
        predictions.append(prediction)
        latencies_ms.append(latency_ms)

        details.append(
            {
                "row_index": int(index),
                "picture_url": row["picture_url"],
                "target_has_infraction": bool(row["target_has_infraction"]),
                "prediction": prediction,
                "is_correct": bool(row["target_has_infraction"]) == prediction,
                "segment": row.get("segment", ""),
                "site": row.get("site", ""),
                "latency_ms": round(latency_ms, 2),
                "score": score,
                "rule_score": rule_score,
                "model_score": model_score,
                "semantic_score": semantic_score,
                "visual_score": visual_score,
                "visual_model_score": visual_model_score,
                "ocr_len": len(ocr_text),
                "ocr_text": ocr_text,
                "evidence": evidence,
                "error": error,
            }
        )

        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(
                f"Processed {index + 1}/{len(frame)} rows; "
                f"failures={failed}; last_latency_ms={latency_ms:.0f}",
                flush=True,
            )

    report = evaluate_predictions(frame, predictions)
    report["dataset_csv"] = str(args.dataset_csv)
    report["evaluated_rows"] = int(len(frame))
    report["dedupe_group_key"] = args.dedupe_group_key
    report["preset"] = args.preset
    report["ocr_backend"] = args.ocr_backend
    report["visual_classifier"] = args.visual_classifier
    report["visual_model_threshold"] = args.visual_model_threshold
    report["visual_model_path"] = str(args.visual_model_path) if args.visual_classifier != "none" else ""
    report["failed_rows"] = int(failed)
    report["failure_rate"] = failed / len(frame) if len(frame) else 0.0
    report["latency_ms"] = {
        "min": min(latencies_ms) if latencies_ms else 0.0,
        "median": median(latencies_ms) if latencies_ms else 0.0,
        "p95": percentile(latencies_ms, 0.95),
        "p99": percentile(latencies_ms, 0.99),
        "max": max(latencies_ms) if latencies_ms else 0.0,
    }
    report["failed_rows_detail"] = [row for row in details if row["error"]]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    details_csv = args.details_csv or args.output_json.with_suffix(".details.csv")
    details_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(details_csv, index=False)

    print(json.dumps(report["overall"], indent=2))
    print(json.dumps(report["latency_ms"], indent=2))
    print(f"Saved report to {args.output_json}")
    print(f"Saved details to {details_csv}")


if __name__ == "__main__":
    main()
