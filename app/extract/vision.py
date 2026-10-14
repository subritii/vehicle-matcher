"""Optional local vision model (Ollama) that reads invoice fields from an image.

The model only *proposes* raw strings; app.extract.invoice.fill_gaps_with_vision
decides what to keep (parsing, VIN gates, sources). Set OLLAMA_MODEL to enable it.

Memory: on an 8 GB machine the OCR worker and the vision model must not be loaded
together. Each request uses ``keep_alive=0`` (Ollama unloads the model right after),
or, for a batch, ``session()`` keeps it loaded and unloads it on exit.
"""

from __future__ import annotations

import base64
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path

import httpx

from app.extract.invoice import FIELD_NAMES

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_HOST = "http://localhost:11434"

# Every field is a string or null; parsing and validation happen in invoice.py / photo.py.
SCHEMA = {
    "type": "object",
    "properties": {name: {"type": ["string", "null"]} for name in FIELD_NAMES},
    "required": list(FIELD_NAMES),
}
PHOTO_FIELDS = ("shot_type", "vin", "plate", "plate_state", "odometer", "color", "body_style")
PHOTO_SHOT_TYPES = ("exterior", "vin_plate", "door_jamb", "odometer", "plate", "interior", "other")
PHOTO_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": ["string", "null"]} for name in PHOTO_FIELDS}
    | {"shot_type": {"type": "string", "enum": list(PHOTO_SHOT_TYPES)}},
    "required": list(PHOTO_FIELDS),
}


class VisionError(RuntimeError):
    """The model gave no usable answer; the rules-only result is kept."""


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text()


class VisionModel:
    def __init__(self, model: str, host: str = DEFAULT_HOST, timeout: float = 300,
                 max_side: int = 1280, client: httpx.Client | None = None) -> None:
        self.model = model
        self.host = host
        self.name = f"vision:{model}"
        self.max_side = max_side
        self.keep_alive: int | str = 0
        self._think = True  # send think=false; dropped for models that reject the flag
        self._client = client or httpx.Client(base_url=host, timeout=timeout)

    def _encode(self, image) -> str:
        image = image.convert("RGB")
        scale = self.max_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def read_invoice(self, images: list) -> dict[str, str | None]:
        return self._ask("invoice_fields", SCHEMA, FIELD_NAMES, images)

    def read_photo(self, images: list) -> dict[str, str | None]:
        """Shot type plus whatever VIN, plate, odometer, color and body style the photo shows."""
        return self._ask("photo_fields", PHOTO_SCHEMA, PHOTO_FIELDS, images)

    def _ask(self, prompt: str, schema: dict, names, images: list) -> dict[str, str | None]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": load_prompt(prompt),
                          "images": [self._encode(im) for im in images]}],
            "format": schema,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        try:
            if self._think:
                r = self._client.post("/api/chat", json={**body, "think": False})
                if r.status_code == 400 and "think" in r.text:
                    self._think = False
            if not self._think:
                r = self._client.post("/api/chat", json=body)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise VisionError(f"{self.model}: {e}") from None
        try:
            data = json.loads(content)
        except ValueError:
            raise VisionError(f"{self.model}: answer is not JSON: {content[:80]!r}") from None
        if not isinstance(data, dict):
            raise VisionError(f"{self.model}: answer is not a JSON object")
        return {k: (None if v is None else str(v)) for k, v in data.items() if k in names}

    def unload(self) -> None:
        try:
            self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass

    @contextmanager
    def session(self, keep_alive: str = "10m"):
        """Keep the model loaded across a batch of images, then unload it."""
        self.keep_alive = keep_alive
        try:
            yield self
        finally:
            self.keep_alive = 0
            self.unload()


def from_env() -> VisionModel | None:
    """The model named by OLLAMA_MODEL, or None (rules + OCR only)."""
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        return None
    return VisionModel(model, host=os.getenv("OLLAMA_HOST", "").strip() or DEFAULT_HOST,
                       timeout=float(os.getenv("OLLAMA_TIMEOUT", "") or 300))
