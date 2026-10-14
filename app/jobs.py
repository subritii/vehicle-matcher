"""Uploads processed in the background, one job and one engine at a time.

A single worker thread runs the jobs in order, so OCR and the vision model never run
together, not even across uploads. Each step is saved as it starts and ends, so the
upload page can show progress.
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app import pipeline
from app.extract.invoice import fill_gaps_with_vision, stop_ocr
from app.extract.photo import extract_photos
from app.match.cluster import cluster_photos
from app.match.score import match
from app.models import Invoice, Job

log = logging.getLogger(__name__)

STEPS = (
    ("save", "Saving your files"),
    ("invoice_text", "Reading the invoice…"),
    ("invoice_ai", "AI double-checks the invoice…"),
    ("photo_ai", "Looking at the photos…"),
    ("photo_ocr", "Reading the VIN and mileage from the photos…"),
    ("photo_alpr", "Reading the license plate…"),
    ("match", "Finding the match…"),
)
PHOTO_STEPS = {"vision": "photo_ai", "ocr": "photo_ocr", "alpr": "photo_alpr"}


def new_steps() -> list[dict]:
    return [{"key": k, "label": label, "status": "pending", "detail": ""} for k, label in STEPS]


class Progress:
    """Updates the job's steps and commits, so the page sees each change."""

    def __init__(self, db, job: Job) -> None:
        self.db, self.job = db, job

    def _set(self, key: str, status: str, detail: str | None = None) -> None:
        steps = [dict(s) for s in self.job.steps_json]
        for s in steps:
            if s["key"] == key:
                s["status"] = status
                if detail is not None:
                    s["detail"] = detail
        self.job.steps_json = steps
        self.db.commit()

    def start(self, key: str) -> None:
        self._set(key, "running")

    def done(self, key: str, detail: str = "") -> None:
        self._set(key, "done", detail)

    def skip(self, key: str, detail: str) -> None:
        self._set(key, "skipped", detail)

    def running(self) -> str | None:
        return next((s["key"] for s in self.job.steps_json if s["status"] == "running"), None)


@dataclass
class Context:
    sessions: object  # sessionmaker
    storage: object  # LocalStorage
    settings: object
    vision_factory: object  # () -> VisionModel | None


def _source_label(source: str) -> str:
    return "read from the PDF" if source == "pdf_text" else "read by OCR" if source.startswith("ocr:") else source


def process_upload(ctx: Context, job_id: int) -> None:
    with ctx.sessions() as db:
        job = db.get(Job, job_id)
        job.status = "running"
        p = Progress(db, job)
        p.done("save", f"1 invoice and {len(job.photo_paths)} photo{'s' * (len(job.photo_paths) != 1)}")
        readers = pipeline.Readers.build(ctx.settings, vision=ctx.vision_factory())
        inv_path = ctx.storage.path(job.invoice_path)

        p.start("invoice_text")
        f = readers.invoice_rules(inv_path)
        p.done("invoice_text", f"Found {len(f.fields)} details ({_source_label(f.text_source)})")
        if readers.vision is None:
            p.skip("invoice_ai", "AI is turned off")
        else:
            p.start("invoice_ai")
            stop_ocr()  # never OCR and the model at once
            before = {k: v.source for k, v in f.fields.items()}
            with readers.vision.session():
                f = fill_gaps_with_vision(f, inv_path, readers.vision, vpic=readers.vpic)
            added = [k for k, v in f.fields.items() if v.source.startswith("vision") and before.get(k) != v.source]
            p.done("invoice_ai", f"AI added {len(added)} detail{'s' * (len(added) != 1)}")
        iid = f"up{job.id:03d}"
        prefix = f"uploads/{job.id}"
        kind = "pdf" if inv_path.suffix.lower() == ".pdf" else "photo"
        pipeline.add_invoice(db, ctx.storage, iid, inv_path, kind, f, readers.ocr_cache, prefix)
        job.invoice_id = iid
        db.commit()

        paths = [ctx.storage.path(k) for k in job.photo_paths]
        fields = []
        if paths:
            def progress(step: str) -> None:
                if (cur := p.running()) is not None:
                    p.done(cur)
                key = PHOTO_STEPS[step]
                if step == "vision" and readers.vision is None:
                    p.skip(key, "AI is turned off")
                else:
                    p.start(key)

            fields = extract_photos(paths, readers.photo_engines(), uploaded_at=job.created_at, vpic=readers.vpic,
                                    progress=progress)
            if (cur := p.running()) is not None:
                p.done(cur)
        else:
            for key in PHOTO_STEPS.values():
                p.skip(key, "No photos")

        p.start("match")
        rows = pipeline.add_photos(db, ctx.storage, prefix,
                                   [(pipeline.PhotoIn(path, job.created_at, job.uploader), pf)
                                    for path, pf in zip(paths, fields)])
        clusters = cluster_photos([obs for _, obs in rows])
        pool = [pipeline.invoice_obs_from_json(i.obs_json)
                for i in db.scalars(select(Invoice).where(Invoice.status != "linked"))]
        decisions = match(clusters, pool, vpic=readers.vpic or None)
        stored = pipeline.apply_match(db, clusters, decisions, readers.vpic, id_prefix=f"u{job.id}-")
        job.cluster_ids = [c.id for c in stored]
        summary = {s: sum(1 for c in stored if c.status == s) for s in ("linked", "review", "unmatched")}
        words = {"linked": "matched", "review": "needs your check", "unmatched": "no invoice found yet"}
        p.done("match", ", ".join(f"{n} photo set{'s' * (n != 1)} {words[s]}" for s, n in summary.items() if n)
               or "No photos to match")
        job.status = "done"
        db.commit()


class JobRunner:
    def __init__(self, ctx: Context, process=process_upload) -> None:
        self.ctx, self.process = ctx, process
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload")

    def submit(self, job_id: int):
        return self.executor.submit(self._run, job_id)

    def _run(self, job_id: int) -> None:
        try:
            self.process(self.ctx, job_id)
        except Exception as e:  # the job shows the error; the worker keeps running
            log.error("upload job %s failed:\n%s", job_id, traceback.format_exc())
            with self.ctx.sessions() as db:
                job = db.get(Job, job_id)
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"[:500]
                steps = [dict(s, status="failed") if s["status"] == "running" else s for s in job.steps_json]
                job.steps_json = steps
                db.commit()


def sample_files(storage) -> tuple[Path, list[Path]] | None:
    """The vehicle tools/seed_demo.py kept out of the demo database for the Upload page."""
    root = storage.path("demo/sample")
    invoices = sorted(root.glob("invoice.*")) if root.exists() else []
    if not invoices:
        return None
    return invoices[0], sorted((root / "photos").glob("*.jpg"))
