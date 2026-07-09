from __future__ import annotations

import io
import hashlib
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DownloadedImage:
    url: str
    content: bytes
    image: Image.Image


def download_image(
    picture_url: str,
    timeout_seconds: float = 5.0,
    verify_ssl: bool = False,
    user_agent: str = "meli-image-moderation-poc/0.1",
) -> DownloadedImage:
    context = None if verify_ssl else ssl._create_unverified_context()
    request = urllib.request.Request(picture_url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as response:
        content = response.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return DownloadedImage(url=picture_url, content=content, image=image)


def image_cache_path(picture_url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(picture_url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.jpg"


def load_or_download_image(
    picture_url: str,
    cache_dir: Path,
    timeout_seconds: float = 5.0,
    verify_ssl: bool = False,
    user_agent: str = "meli-image-moderation-poc/0.1",
) -> DownloadedImage:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = image_cache_path(picture_url, cache_dir)
    if path.exists():
        content = path.read_bytes()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        return DownloadedImage(url=picture_url, content=content, image=image)

    downloaded = download_image(
        picture_url,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
        user_agent=user_agent,
    )
    downloaded.image.save(path, format="JPEG", quality=92)
    return downloaded
