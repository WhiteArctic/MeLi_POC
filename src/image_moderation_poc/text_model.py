from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_moderation_poc.policy import normalize_text


def iter_features(text: str, char_ngrams: tuple[int, ...] = (3, 4, 5)) -> list[int]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    padded = f" {normalized} "
    features: list[int] = []
    for token in normalized.split():
        if len(token) >= 2:
            features.append(_hash_feature(f"w:{token}"))
    for n in char_ngrams:
        if len(padded) >= n:
            for i in range(0, len(padded) - n + 1):
                features.append(_hash_feature(f"c{n}:{padded[i:i+n]}"))
    return features


def _hash_feature(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


@dataclass
class TextNaiveBayes:
    feature_count: int
    class_log_prior: np.ndarray
    feature_log_prob: np.ndarray
    alpha: float = 1.0

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[bool],
        feature_count: int = 2**18,
        alpha: float = 1.0,
    ) -> "TextNaiveBayes":
        counts = np.zeros((2, feature_count), dtype=np.float64)
        class_counts = np.zeros(2, dtype=np.float64)
        for text, label in zip(texts, labels, strict=True):
            class_index = 1 if label else 0
            class_counts[class_index] += 1
            for feature in iter_features(text):
                counts[class_index, feature % feature_count] += 1.0

        smoothed = counts + alpha
        feature_log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        class_log_prior = np.log((class_counts + alpha) / (class_counts.sum() + 2 * alpha))
        return cls(feature_count, class_log_prior, feature_log_prob, alpha)

    def predict_proba_one(self, text: str) -> float:
        features = iter_features(text)
        if not features:
            return float(math.exp(self.class_log_prior[1]))
        indexes = np.asarray([feature % self.feature_count for feature in features], dtype=np.int64)
        scores = self.class_log_prior.copy()
        scores[0] += self.feature_log_prob[0, indexes].sum()
        scores[1] += self.feature_log_prob[1, indexes].sum()
        max_score = float(np.max(scores))
        probs = np.exp(scores - max_score)
        probs = probs / probs.sum()
        return float(probs[1])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_count=np.asarray([self.feature_count], dtype=np.int64),
            class_log_prior=self.class_log_prior,
            feature_log_prob=self.feature_log_prob,
            alpha=np.asarray([self.alpha], dtype=np.float64),
        )
        metadata = {
            "model_type": "hashed_char_word_multinomial_naive_bayes",
            "feature_count": self.feature_count,
            "alpha": self.alpha,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TextNaiveBayes":
        data = np.load(path)
        return cls(
            feature_count=int(data["feature_count"][0]),
            class_log_prior=data["class_log_prior"],
            feature_log_prob=data["feature_log_prob"],
            alpha=float(data["alpha"][0]),
        )

