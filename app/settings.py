"""Settings from the environment (and .env): data folder, database and demo mode."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


DEMO_MODES = ("full", "showcase")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    # full: the whole pipeline. showcase: the read-only public demo served from a prebuilt
    # bundle (showcase_dir, written by tools/build_showcase.py); data_dir then only holds the
    # visitors' session databases.
    demo_mode: str = "full"
    showcase_dir: Path = ROOT / "showcase"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "showcase_dir", Path(self.showcase_dir))

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
    mode = os.getenv("DEMO_MODE", "").strip().lower() or "full"
    showcase = Path(os.getenv("SHOWCASE_DIR") or ROOT / "showcase")
    return Settings(d, url, mode, showcase if showcase.is_absolute() else ROOT / showcase)
