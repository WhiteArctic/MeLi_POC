from __future__ import annotations

import json
import math
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def extract_visual_features(image: Image.Image, size: int = 224) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = ImageOps.contain(image, (size, size)).convert("RGB")
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))

    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    gray_image = canvas.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    hsv = np.asarray(canvas.convert("HSV"), dtype=np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    edges = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    dark = gray < 0.22
    bright = gray > 0.85
    saturated = (saturation > 0.38) & (value > 0.32)

    features: list[float] = [
        image.width / max(image.height, 1),
        image.height / max(image.width, 1),
        float(gray.mean()),
        float(gray.std()),
        float(edges.mean()),
        float((edges > 0.18).mean()),
        float(dark.mean()),
        float(bright.mean()),
        float(saturated.mean()),
        float(saturation.mean()),
        float(saturation.std()),
    ]

    for channel in range(3):
        values = rgb[:, :, channel]
        features.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(np.quantile(values, 0.10)),
                float(np.quantile(values, 0.50)),
                float(np.quantile(values, 0.90)),
            ]
        )

    for grid in (2, 4):
        cell_h = size // grid
        cell_w = size // grid
        for y in range(grid):
            for x in range(grid):
                ys = slice(y * cell_h, (y + 1) * cell_h)
                xs = slice(x * cell_w, (x + 1) * cell_w)
                features.extend(
                    [
                        float(gray[ys, xs].std()),
                        float((edges[ys, xs] > 0.18).mean()),
                        float(saturated[ys, xs].mean()),
                        float(dark[ys, xs].mean()),
                    ]
                )

    hist_gray, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=True)
    hist_sat, _ = np.histogram(saturation, bins=12, range=(0.0, 1.0), density=True)
    features.extend(hist_gray.astype(float).tolist())
    features.extend(hist_sat.astype(float).tolist())
    return np.asarray(features, dtype=np.float32)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale


@dataclass
class VisualFeatureClassifier:
    weights: np.ndarray
    bias: float
    scaler: FeatureScaler
    threshold: float = 0.90

    @classmethod
    def train(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 350,
        learning_rate: float = 0.08,
        l2: float = 0.002,
        threshold: float = 0.90,
    ) -> "VisualFeatureClassifier":
        scaler = FeatureScaler.fit(features)
        x = scaler.transform(features)
        y = labels.astype(np.float32)
        weights = np.zeros(x.shape[1], dtype=np.float32)
        bias = 0.0
        positive_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))

        for _ in range(epochs):
            logits = x @ weights + bias
            probs = _sigmoid(logits)
            sample_weights = np.where(y > 0.5, positive_weight, 1.0).astype(np.float32)
            error = (probs - y) * sample_weights
            grad_w = (x.T @ error) / len(y) + l2 * weights
            grad_b = float(error.mean())
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        return cls(weights=weights, bias=bias, scaler=scaler, threshold=threshold)

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(features)
        return _sigmoid(x @ self.weights + self.bias)

    def predict_proba_image(self, image: Image.Image) -> float:
        features = extract_visual_features(image)[None, :]
        return float(self.predict_proba_features(features)[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float32),
            mean=self.scaler.mean,
            scale=self.scaler.scale,
            threshold=np.asarray([self.threshold], dtype=np.float32),
        )
        metadata = {
            "model_type": "visual_feature_logistic_regression",
            "feature_count": int(self.weights.shape[0]),
            "threshold": self.threshold,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VisualFeatureClassifier":
        data = np.load(path)
        return cls(
            weights=data["weights"],
            bias=float(data["bias"][0]),
            scaler=FeatureScaler(mean=data["mean"], scale=data["scale"]),
            threshold=float(data["threshold"][0]),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class MobileNetV3VisualClassifier:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    threshold: float = 0.91
    device: str | None = None
    _runtime: tuple[Any, Any, Any] | None = None

    @classmethod
    def is_available(cls) -> bool:
        return find_spec("torch") is not None and find_spec("torchvision") is not None

    @classmethod
    def load(cls, path: Path, threshold: float | None = None) -> "MobileNetV3VisualClassifier":
        if not path.exists():
            raise FileNotFoundError(f"Visual model not found: {path}")
        data = np.load(path)
        stored_threshold = float(data["threshold"][0]) if "threshold" in data.files else 0.91
        return cls(
            weights=data["weights"],
            bias=float(data["bias"][0]),
            mean=data["mean"],
            scale=data["scale"],
            threshold=stored_threshold if threshold is None else threshold,
        )

    def predict_proba_image(self, image: Image.Image) -> float:
        torch, transform, backbone = self._load_runtime()
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        tensor = transform(prepared).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = backbone(tensor).cpu().numpy().astype(np.float32)
        x = (embedding - self.mean) / self.scale
        probability = _sigmoid(x @ self.weights + self.bias)
        return float(probability[0])

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if not self.is_available():
            raise RuntimeError("torch and torchvision are required for MobileNetV3VisualClassifier")

        import torch
        from torch import nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        device = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        model_weights = MobileNet_V3_Small_Weights.DEFAULT
        transform = model_weights.transforms()
        model = mobilenet_v3_small(weights=model_weights)
        backbone = nn.Sequential(model.features, model.avgpool, nn.Flatten()).to(device).eval()
        self._runtime = (torch, transform, backbone)
        return self._runtime
