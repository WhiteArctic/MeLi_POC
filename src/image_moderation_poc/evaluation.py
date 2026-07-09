from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> Metrics:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return Metrics(tp, fp, tn, fn, precision, recall, f1, accuracy)


def evaluate_predictions(frame: pd.DataFrame, predictions: list[bool]) -> dict[str, object]:
    metrics = compute_metrics(frame["target_has_infraction"].astype(bool).tolist(), predictions)
    by_segment = {}
    for segment, segment_frame in frame.groupby("segment", dropna=False):
        indexes = segment_frame.index.tolist()
        segment_predictions = [predictions[i] for i in indexes]
        by_segment[str(segment)] = compute_metrics(
            segment_frame["target_has_infraction"].astype(bool).tolist(),
            segment_predictions,
        ).as_dict()
    return {"overall": metrics.as_dict(), "by_segment": by_segment}


def save_evaluation_report(report: dict[str, object], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

