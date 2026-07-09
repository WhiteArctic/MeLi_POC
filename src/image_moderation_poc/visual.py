from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from image_moderation_poc.schemas import EvidenceMatch


@dataclass
class VisualHeuristicDetector:
    max_side: int = 384

    def detect(self, image: Image.Image) -> list[EvidenceMatch]:
        prepared = self._prepare(image)
        edge_density = self._edge_density(prepared)
        contrast = self._contrast(prepared)
        saturated_ratio = self._saturated_ratio(image)

        matches: list[EvidenceMatch] = []
        if edge_density >= 0.18 and contrast >= 42:
            score = min(0.45, 0.25 + edge_density * 0.75 + min(contrast, 90) / 450)
            matches.append(
                EvidenceMatch(
                    "visual_overlay_signal",
                    "high contrast text or badge overlay",
                    f"edge_density={edge_density:.2f}, contrast={contrast:.0f}",
                    score,
                    "visual_heuristic",
                )
            )

        if saturated_ratio >= 0.18 and edge_density >= 0.12:
            score = min(0.35, 0.18 + saturated_ratio * 0.6)
            matches.append(
                EvidenceMatch(
                    "visual_overlay_signal",
                    "saturated promotional badge-like regions",
                    f"saturated_ratio={saturated_ratio:.2f}, edge_density={edge_density:.2f}",
                    score,
                    "visual_heuristic",
                )
            )

        return sorted(matches, key=lambda m: m.score, reverse=True)

    def _prepare(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        scale = min(1.0, self.max_side / max(width, height))
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        return ImageOps.autocontrast(image)

    def _edge_density(self, image: Image.Image) -> float:
        edges = image.filter(ImageFilter.FIND_EDGES)
        values = np.asarray(edges, dtype=np.uint8)
        return float((values > 42).mean())

    def _contrast(self, image: Image.Image) -> float:
        values = np.asarray(image, dtype=np.float32)
        return float(values.std())

    def _saturated_ratio(self, image: Image.Image) -> float:
        hsv = ImageOps.exif_transpose(image).convert("HSV")
        values = np.asarray(hsv, dtype=np.uint8)
        saturation = values[:, :, 1]
        value = values[:, :, 2]
        return float(((saturation > 95) & (value > 80)).mean())
