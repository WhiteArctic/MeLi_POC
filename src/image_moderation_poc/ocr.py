from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class OCRBackend:
    name = "base"

    def extract_text(self, image: Image.Image) -> str:
        raise NotImplementedError


class NullOCR(OCRBackend):
    name = "null"

    def extract_text(self, image: Image.Image) -> str:
        return ""


@dataclass
class EnsembleOCR(OCRBackend):
    backends: tuple[OCRBackend, ...]
    name: str = "ensemble"

    def extract_text(self, image: Image.Image) -> str:
        chunks: list[str] = []
        seen: set[str] = set()
        for backend in self.backends:
            text = backend.extract_text(image)
            for line in text.splitlines():
                normalized = " ".join(line.split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                chunks.append(line.strip())
        return "\n".join(chunks)


@dataclass
class EasyOCRBackend(OCRBackend):
    langs: tuple[str, ...] = ("es", "en", "pt")
    gpu: bool = False
    name: str = "easyocr"
    _reader: object | None = None

    def is_available(self) -> bool:
        return find_spec("easyocr") is not None

    def extract_text(self, image: Image.Image) -> str:
        reader = self._get_reader()
        array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        results = reader.readtext(array, detail=0, paragraph=False)
        return "\n".join(str(result).strip() for result in results if str(result).strip())

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("EasyOCR is not installed in this Python environment") from exc
            self._reader = easyocr.Reader(list(self.langs), gpu=self.gpu, verbose=False)
        return self._reader


@dataclass
class TesseractCliOCR(OCRBackend):
    lang: str = "spa+eng+por"
    timeout_seconds: float = 4.0
    pass_timeout_seconds: float = 1.25
    pass_plan: tuple[tuple[str, str], ...] = (
        ("original", "6"),
        ("gray_2x", "6"),
        ("contrast_2x", "11"),
        ("binary_2x", "11"),
    )
    name: str = "tesseract_cli"

    def is_available(self) -> bool:
        return shutil.which("tesseract") is not None

    def extract_text(self, image: Image.Image) -> str:
        if not self.is_available():
            return ""

        deadline = time.monotonic() + self.timeout_seconds
        chunks: list[str] = []
        seen_lines: set[str] = set()

        for variant, psm in self.pass_plan:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.05:
                break

            pass_timeout = min(self.pass_timeout_seconds, remaining_seconds)
            pass_image = self._preprocess(image, variant)
            text = self._run_tesseract(pass_image, psm=psm, timeout_seconds=pass_timeout)
            if not text:
                continue

            new_lines = []
            for line in text.splitlines():
                normalized = " ".join(line.split())
                if not normalized or normalized in seen_lines:
                    continue
                seen_lines.add(normalized)
                new_lines.append(line.strip())
            if new_lines:
                chunks.append("\n".join(new_lines))

        return "\n".join(chunks).strip()

    def _run_tesseract(self, image: Image.Image, psm: str, timeout_seconds: float) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image.save(image_file.name)
            command = [
                "tesseract",
                image_file.name,
                "stdout",
                "-l",
                self.lang,
                "--psm",
                psm,
                "-c",
                "preserve_interword_spaces=1",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    def _preprocess(self, image: Image.Image, variant: str) -> Image.Image:
        if variant == "original":
            return image.convert("RGB")

        gray = image.convert("L")
        if variant == "gray_2x":
            return self._resize(gray, 2.0)

        resized = self._resize(gray, 2.0)
        contrasted = ImageOps.autocontrast(resized).filter(ImageFilter.SHARPEN)
        if variant == "contrast_2x":
            return contrasted

        if variant == "binary_2x":
            return contrasted.point(lambda pixel: 255 if pixel > 160 else 0, mode="1")

        return image.convert("RGB")

    def _resize(self, image: Image.Image, factor: float) -> Image.Image:
        width, height = image.size
        return image.resize(
            (max(1, int(width * factor)), max(1, int(height * factor))),
            Image.Resampling.LANCZOS,
        )
