from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_moderation_poc.datasets import POLICY_KEYWORD_RE
from image_moderation_poc.evaluation import compute_metrics, evaluate_predictions, save_evaluation_report
from image_moderation_poc.text_model import TextNaiveBayes


def main() -> None:
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "datasets" / "validation.csv"
    output = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "outputs" / "reports" / f"{dataset.stem}_baseline_comparison.json"
    )
    model = TextNaiveBayes.load(ROOT / "outputs" / "models" / "text_nb_model_v1.npz")
    frame = pd.read_csv(dataset).reset_index(drop=True)
    y_true = frame["target_has_infraction"].astype(bool).tolist()
    texts = frame["ocr_text"].fillna("").astype(str).tolist()
    regex_predictions = frame["ocr_text"].fillna("").astype(str).str.contains(
        POLICY_KEYWORD_RE, regex=True
    ).tolist()
    probabilities = [model.predict_proba_one(text) for text in texts]

    comparison: dict[str, object] = {
        "dataset_csv": str(dataset),
        "rows": len(frame),
        "baselines": {},
    }
    comparison["baselines"]["regex_policy_keyword"] = compute_metrics(
        y_true, regex_predictions
    ).as_dict()
    for threshold in [0.40, 0.58, 0.80, 0.95, 0.99]:
        model_predictions = [prob >= threshold for prob in probabilities]
        ensemble_predictions = [
            regex_hit or prob >= threshold
            for regex_hit, prob in zip(regex_predictions, probabilities, strict=True)
        ]
        comparison["baselines"][f"model_only_threshold_{threshold:.2f}"] = compute_metrics(
            y_true, model_predictions
        ).as_dict()
        comparison["baselines"][f"regex_or_model_threshold_{threshold:.2f}"] = compute_metrics(
            y_true, ensemble_predictions
        ).as_dict()

    save_evaluation_report(comparison, output)
    print(json.dumps(comparison["baselines"], indent=2))


if __name__ == "__main__":
    main()

