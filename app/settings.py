"""Settings from the environment (and .env): data folder and database."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str

    @property
    def media_dir(self) -> Path:
        """Everything the web app may serve: display images, thumbnails, uploads."""
        return self.data_dir / "media"

    @property
    def ocr_cache(self) -> Path:
        return self.data_dir / "ocr_cache"

    @property
    def extract_cache(self) -> Path:
        return self.data_dir / "extract_cache"


def get_settings(data_dir: str | Path | None = None, database_url: str | None = None) -> Settings:
    d = Path(data_dir or os.getenv("DATA_DIR") or "data")
    if not d.is_absolute():
        d = ROOT / d
    url = database_url or os.getenv("DATABASE_URL") or f"sqlite:///{d / 'app.db'}"
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = Path(url.removeprefix("sqlite:///"))  # relative to the repo, not the cwd
        url = f"sqlite:///{ROOT / rel}" if not rel.is_absolute() else url
    return Settings(d, url)
