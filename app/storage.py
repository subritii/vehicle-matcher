"""File storage behind a small interface (local folder now, S3 later).

Paths are relative keys like ``demo/invoices/inv_001.jpg``; the web app serves them
under ``/media/``. Display images and thumbnails are resized JPEGs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

DISPLAY_SIDE = 1400
THUMB_SIDE = 360


def safe_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "file"
    return base[:80]


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"outside storage: {key}")
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def put_file(self, key: str, src: str | Path) -> str:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, p)
        return key

    def put_image(self, key: str, image, max_side: int) -> str:
        """Save a PIL image as JPEG, scaled down so its long side is at most ``max_side``."""
        im = image.convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)))
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, "JPEG", quality=85, optimize=True)
        return key

    def delete_prefix(self, prefix: str) -> None:
        p = self.path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
