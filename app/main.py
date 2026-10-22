"""Web app: overview, vehicles, review queue, find invoice from photo, results and upload.

Run: ``uvicorn app.main:app --reload`` (after ``python tools/seed_demo.py`` for demo data).
Only this computer and the local network may connect (see ``is_local``); set
ALLOW_PUBLIC_CLIENTS=1 to lift that.

DEMO_MODE=showcase serves the read-only public demo instead (see app/showcase.py): anyone
may connect, with a rate limit, and nothing heavy is loaded.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from sqlalchemy import func, select

from app import jobs, lookup, results, services, showcase
from app.extract.invoice import IMAGE_SUFFIXES
from app.extract.vision import from_env as vision_from_env
from app.models import Cluster, Invoice, Job, make_sessionmaker
from app.settings import DEMO_MODES, Settings, get_settings
from app.storage import LocalStorage, safe_name

HERE = Path(__file__).parent
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_PHOTOS = 40
PAGE = 60  # cards per page on long lists
DOCKER_URL = "https://github.com/subritii/vehicle-matcher#run-it-yourself"

SOURCE_BADGES = {"pdf_text": ("Read from PDF", "pdf"), "vin": ("From VIN lookup", "vin"),
                 "exif": ("From photo info", "pdf"), "barcode": ("Read from barcode", "pdf"),
                 "alpr": ("Read by plate reader", "ocr"), "rules": ("Rules", "pdf"), "upload": ("Upload time", "pdf")}


def source_badge(source: str | None) -> tuple[str, str]:
    """(plain label, color key) for where a field's value came from."""
    s = source or ""
    if s.startswith("ocr:"):
        return "Read by OCR", "ocr"
    if s.startswith("vision"):
        return "Read by AI", "ai"
    return SOURCE_BADGES.get(s, (s or "-", "pdf"))


def _longdate(v) -> str:
    """2025-12-24 or a date/datetime -> Dec 24, 2025."""
    from datetime import date as _date

    if not v:
        return ""
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v)
        except ValueError:
            return v
    return v.strftime("%b %d, %Y") if isinstance(v, (_date, datetime)) else str(v)


def vin_diff(a: str | None, b: str | None) -> Markup:
    """``a`` with the characters that differ from ``b`` highlighted."""
    if not a:
        return Markup("")
    if not b or len(a) != len(b) or sum(x != y for x, y in zip(a, b)) > 5:
        return escape(a)  # only near-misses get their differences marked
    return Markup("".join(str(escape(x)) if x == y else f'<mark class="diff">{escape(x)}</mark>' for x, y in zip(a, b)))


def is_local(host: str | None) -> bool:
    """This computer or the local network: loopback, private (10/8, 172.16/12, 192.168/16,
    fc00::/7) and link-local addresses. Anything else, e.g. a request forwarded from the
    internet, is refused."""
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_loopback or ip.is_link_local or ip.is_private


def create_app(settings: Settings | None = None, process=None, vision_factory=None, read_photos=None,
               local_only: bool | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.demo_mode not in DEMO_MODES:
        raise ValueError(f"DEMO_MODE must be one of {', '.join(DEMO_MODES)}, not {settings.demo_mode!r}")
    is_showcase = settings.demo_mode == "showcase"
    if is_showcase:
        bundle = showcase.Bundle.load(settings.showcase_dir)
        visitors = showcase.VisitorDatabases(bundle.db_path, settings.data_dir / "visitors")
        sessions = showcase.Sessions(visitors)
        media_dir = bundle.media_dir
        vision_factory = lambda: None  # noqa: E731  no model, no OCR, no plate reader
        local_only = False
    else:
        bundle = None
        media_dir = settings.media_dir
        media_dir.mkdir(parents=True, exist_ok=True)
        sessions = make_sessionmaker(settings.database_url)
    storage = LocalStorage(media_dir)
    vision_factory = vision_factory or vision_from_env
    ctx = jobs.Context(sessions, storage, settings, vision_factory)
    runner = jobs.JobRunner(ctx, process or jobs.process_upload)
    lookups = lookup.LookupStore()
    lookup_dir = settings.data_dir / "lookups"
    shutil.rmtree(lookup_dir, ignore_errors=True)  # lookups live in memory: older files are orphans
    lookup_dir.mkdir(parents=True)
    read_photos = read_photos or lookup.default_reader(settings, vision_factory)
    if local_only is None:
        local_only = os.getenv("ALLOW_PUBLIC_CLIENTS", "").strip() != "1"

    app = FastAPI(title="Vehicle Matcher")
    app.state.sessions, app.state.storage, app.state.runner = sessions, storage, runner
    app.state.lookups, app.state.lookup_dir = lookups, lookup_dir

    if local_only:
        @app.middleware("http")
        async def local_network_only(request: Request, call_next):
            if not is_local(request.client.host if request.client else None):
                return PlainTextResponse("This demo only answers this computer and the local network.", 403)
            return await call_next(request)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.mount("/media", StaticFiles(directory=media_dir), name="media")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(source_badge=source_badge, vin_diff=vin_diff, plain=services.plain_reason,
                                 plain_reasons=services.plain_reasons, match_sentence=services.match_sentence)
    templates.env.filters["make"] = lambda m: (m.upper() if len(m) <= 3 else m.title()) if m else m  # BMW, Ford
    templates.env.filters["longdate"] = _longdate
    templates.env.filters["sentence"] = lambda t: t[:1].upper() + t[1:] if t else t  # unlike capitalize: keeps "VIN"
    templates.env.tests["startswith"] = lambda s, prefix: str(s).startswith(prefix)
    templates.env.filters["money"] = lambda v: f"${float(v):,.2f}" if v not in (None, "") else "-"
    templates.env.globals.update(showcase=is_showcase, docker_url=os.getenv("SHOWCASE_DOCKER_URL") or DOCKER_URL)

    def render(request: Request, name: str, status_code: int = 200, **ctx) -> HTMLResponse:
        with sessions() as db:
            ctx.setdefault("review_count", services.review_count(db))
        ctx.setdefault("active", name.split(".")[0].split("_")[0])
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    def go(request: Request, url: str) -> Response:
        if request.headers.get("HX-Request"):
            return Response(status_code=204, headers={"HX-Location": url})
        return RedirectResponse(url, status_code=303)

    if is_showcase:
        pages = showcase.RateLimiter(showcase.limit_from_env("SHOWCASE_RATE_LIMIT", showcase.DEFAULT_PAGE_LIMIT))
        posts = showcase.RateLimiter(showcase.limit_from_env("SHOWCASE_POST_LIMIT", showcase.DEFAULT_POST_LIMIT))

        @app.middleware("http")
        async def showcase_guard(request: Request, call_next):
            """Rate limits, nothing but review decisions written, and each visitor's decisions in
            their own copy of the database."""
            path, ip = request.url.path, request.client.host if request.client else "-"
            slow_down = PlainTextResponse("Too many requests. Please wait a minute and try again.", 429,
                                          headers={"Retry-After": "60"})
            if showcase.counted(path) and not pages.allow(ip):
                return slow_down
            if request.method not in ("GET", "HEAD"):
                if not posts.allow(ip):
                    return slow_down
                size = request.headers.get("content-length")
                if size is None or not size.isdigit() or int(size) > showcase.MAX_BODY:
                    return PlainTextResponse("Request too large.", 413)
                if not showcase.REVIEW_POST.match(path):
                    return render(request, "showcase_off.html", status_code=403, active="")
            own = visitors.get(request.cookies.get(showcase.COOKIE))
            new = None
            if own is None and request.method == "POST":
                new = visitors.create()
                own = visitors.get(new)
            token = showcase.current_visitor.set(own)
            try:
                response = await call_next(request)
            finally:
                showcase.current_visitor.reset(token)
            if new:
                https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
                response.set_cookie(showcase.COOKIE, new, max_age=int(showcase.VISITOR_IDLE.total_seconds()),
                                    httponly=True, samesite="lax", secure=https)
            return response

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"status": "ok", "mode": settings.demo_mode})

    # --- Overview ---------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request):
        with sessions() as db:
            stats = services.overview(db)
        return render(request, "overview.html", stats=stats, empty=stats["clusters"] == 0)

    # --- Vehicles ---------------------------------------------------------------------

    @app.get("/vehicles", response_class=HTMLResponse)
    def vehicles(request: Request, show: str = "linked", limit: int = PAGE):
        limit = max(PAGE, min(limit, 5000))
        with sessions() as db:
            linked = services.linked_vehicles(db)
            no_photos_count = db.scalar(select(func.count()).select_from(Invoice).where(Invoice.status != "linked"))
            no_photos = services.invoices_without_photos(db, limit=limit) if show == "no_photos" else []
            no_invoice = services.sets_without_invoice(db)
            for c in linked + no_invoice:
                _ = c.photos, c.invoice, c.vehicle  # load before the session closes
            return render(request, "vehicles.html", show=show, linked=linked, no_photos=no_photos,
                          no_photos_count=no_photos_count, no_invoice=no_invoice, limit=limit, page=PAGE)

    @app.get("/vehicles/{invoice_id}", response_class=HTMLResponse)
    def vehicle(request: Request, invoice_id: str):
        with sessions() as db:
            inv = db.get(Invoice, invoice_id)
            if inv is None:
                raise HTTPException(404, "invoice not found")
            cluster = db.scalar(select(Cluster).where(Cluster.invoice_id == invoice_id))
            photos = list(cluster.photos) if cluster else []
            events = services.events_for(db, cluster.id) if cluster else []
            _ = inv.vehicle
            return render(request, "vehicle.html", inv=inv, cluster=cluster, photos=photos, events=events,
                          active="vehicles")

    @app.get("/sets/{cid}", response_class=HTMLResponse)
    def photo_set(request: Request, cid: str):
        with sessions() as db:
            cluster = db.get(Cluster, cid)
            if cluster is None:
                raise HTTPException(404, "photo set not found")
            if cluster.status == "review":
                return RedirectResponse(f"/review/{cid}", status_code=303)
            if cluster.invoice_id:
                return RedirectResponse(f"/vehicles/{cluster.invoice_id}", status_code=303)
            photos = list(cluster.photos)
            return render(request, "photo_set.html", cluster=cluster, photos=photos,
                          events=services.events_for(db, cid), active="vehicles")

    # --- Review -----------------------------------------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    def review_start(request: Request, done: str | None = None):
        with sessions() as db:
            queue = services.review_queue(db)
            undo = services.undo_banner(db, done)
        if queue:
            return RedirectResponse(f"/review/{queue[0].id}" + (f"?done={done}" if done else ""), status_code=303)
        return render(request, "review_empty.html", active="review", undo=undo)

    def _next_url(db, after: str) -> str:
        """The next item, with ?done= so it offers Undo for the decision just made."""
        queue = [c.id for c in services.review_queue(db)]
        later = [c for c in queue if c > after]
        return (f"/review/{(later or queue)[0]}" if queue else "/review") + f"?done={after}"

    @app.get("/review/{cid}", response_class=HTMLResponse)
    def review_item(request: Request, cid: str, error: str | None = None, done: str | None = None):
        with sessions() as db:
            cluster = db.get(Cluster, cid)
            if cluster is None:
                raise HTTPException(404, "photo set not found")
            if cluster.status != "review":
                return RedirectResponse(f"/sets/{cid}", status_code=303)
            queue = [c.id for c in services.review_queue(db)]
            pos = queue.index(cid)
            cands = []
            for cand in cluster.candidates_json[:3]:
                inv = db.get(Invoice, cand["invoice_id"])
                if inv is not None:
                    cands.append({"c": cand, "inv": inv, "rows": services.compare_rows(cluster, inv, cand),
                                  "taken": inv.status == "linked"})
            shown = {c["inv"].id for c in cands}
            others = [i for i in services.unlinked_invoices(db) if i.id not in shown]
            photos = list(cluster.photos)
            undo = services.undo_banner(db, done) if done != cid else None
            return render(request, "review.html", cluster=cluster, photos=photos, cands=cands, others=others, undo=undo,
                          pos=pos, total=len(queue), prev_id=queue[pos - 1] if pos > 0 else None,
                          next_id=queue[pos + 1] if pos + 1 < len(queue) else None, error=error,
                          status_code=409 if error else 200)

    def _act(request: Request, cid: str, fn) -> Response:
        with sessions() as db:
            try:
                fn(db)
                db.commit()
            except services.ActionError as e:
                db.rollback()
                if db.get(Cluster, cid) is not None and db.get(Cluster, cid).status == "review":
                    return review_item(request, cid, error=str(e))
                raise HTTPException(409, str(e)) from None
            return go(request, _next_url(db, cid))

    @app.post("/review/{cid}/confirm")
    def review_confirm(request: Request, cid: str, invoice_id: str = Form(...)):
        return _act(request, cid, lambda db: services.review_link(db, cid, invoice_id, "confirm"))

    @app.post("/review/{cid}/pick")
    def review_pick(request: Request, cid: str, invoice_id: str = Form(...)):
        return _act(request, cid, lambda db: services.review_link(db, cid, invoice_id, "pick"))

    @app.post("/review/{cid}/no-invoice")
    def review_no_invoice(request: Request, cid: str):
        return _act(request, cid, lambda db: services.review_no_invoice(db, cid))

    @app.post("/review/{cid}/undo")
    def review_undo(request: Request, cid: str):
        with sessions() as db:
            try:
                services.review_undo(db, cid)
                db.commit()
            except services.ActionError as e:
                db.rollback()
                raise HTTPException(409, str(e)) from None
        return go(request, f"/review/{cid}")

    # --- Find invoice from photo ------------------------------------------------------

    @app.get("/find", response_class=HTMLResponse)
    def find_form(request: Request, error: str | None = None):
        if bundle is not None:
            return render(request, "find_showcase.html", gallery=bundle.gallery(), meta=bundle.meta, active="find")
        return render(request, "find.html", error=error, vision=vision_factory() is not None,
                      accepted=", ".join(sorted(set(lookup.ACCEPTED.values()))), max_photos=lookup.MAX_PHOTOS,
                      status_code=400 if error else 200)

    @app.post("/find")
    async def find_start(request: Request, photos: list[UploadFile] = File(...)):
        photos = [p for p in photos if p.filename]
        if not photos:
            return find_form(request, error="Choose at least one photo.")
        if any(Path(p.filename).suffix.lower() not in lookup.ACCEPTED for p in photos):
            return find_form(request, error="Photos must be JPG, PNG or HEIC.")
        if len(photos) > lookup.MAX_PHOTOS:
            return find_form(request, error=f"At most {lookup.MAX_PHOTOS} photos at a time.")
        blobs = [await p.read() for p in photos]
        if any(not b or len(b) > MAX_UPLOAD_BYTES for b in blobs):
            return find_form(request, error="Empty photo, or a photo over 40 MB.")
        item = lookups.create(lookup_dir, [p.filename for p in photos])
        for path, data in zip(item.files, blobs):
            path.write_bytes(data)
        runner.executor.submit(lookup.run_lookup, item, read_photos, sessions, vision_factory() is not None)
        return go(request, f"/find/{item.token}")

    def _lookup(token: str) -> lookup.Lookup:
        if bundle is not None:
            if token not in bundle.samples:
                raise HTTPException(404, "No such sample photo.")
            return bundle.samples[token].lookup()
        item = lookups.get(token)
        if item is None:
            raise HTTPException(404, "This lookup has expired or was saved. Start a new one.")
        return item

    def _lookup_ctx(item: lookup.Lookup) -> dict:
        ctx = {"item": item, "invoices": {}, "elapsed": int((datetime.now() - item.created).total_seconds()),
               "sample": bundle.samples[item.token] if bundle is not None else None,
               "meta": bundle.meta if bundle is not None else {}}
        if item.result:
            with sessions() as db:
                ctx["invoices"] = {x["invoice_id"]: db.get(Invoice, x["invoice_id"]) for x in item.result["items"]}
        return ctx

    @app.get("/find/{token}", response_class=HTMLResponse)
    def find_page(request: Request, token: str):
        return render(request, "find_result.html", active="find", **_lookup_ctx(_lookup(token)))

    @app.get("/find/{token}/status", response_class=HTMLResponse)
    def find_status(request: Request, token: str):
        return render(request, "_find_status.html", active="find", **_lookup_ctx(_lookup(token)))

    @app.get("/find/{token}/thumb/{i}.jpg")
    def find_thumb(token: str, i: int):
        path = _lookup(token).thumb(i)
        if not path.exists():
            raise HTTPException(404, "no thumbnail yet")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/find/{token}/save")
    def find_save(request: Request, token: str, invoice_id: str = Form(...)):
        item = _lookup(token)
        if item.status != "done":
            raise HTTPException(409, "The lookup is still running.")
        with sessions() as db:
            try:
                lookup.save(db, storage, item, invoice_id)
                db.commit()
            except services.ActionError as e:
                db.rollback()
                raise HTTPException(409, str(e)) from None
        lookups.drop(token)
        return go(request, f"/vehicles/{invoice_id}")

    # --- Results ----------------------------------------------------------------------

    @app.get("/results", response_class=HTMLResponse)
    def results_page(request: Request):
        return render(request, "results.html", r=results.load())

    if bundle is not None:
        @app.get("/credits", response_class=HTMLResponse)
        def credits_page(request: Request):
            return render(request, "credits.html", credits=bundle.credits, active="")

    # --- Upload -----------------------------------------------------------------------

    @app.get("/upload", response_class=HTMLResponse)
    def upload_form(request: Request, error: str | None = None):
        with sessions() as db:
            recent = db.scalars(select(Job).order_by(Job.id.desc()).limit(5)).all()
        return render(request, "upload.html", recent=recent, error=error, sample=jobs.sample_files(storage),
                      vision=ctx.vision_factory() is not None, status_code=400 if error else 200)

    def _start_job(uploader: str, save) -> int:
        with sessions() as db:
            job = Job(steps_json=jobs.new_steps(), invoice_path="", photo_paths=[],
                      uploader=(uploader or "Demo user").strip()[:60])
            db.add(job)
            db.flush()
            job.invoice_path, job.photo_paths = save(f"uploads/{job.id}")
            db.commit()
            job_id = job.id
        runner.submit(job_id)
        return job_id

    @app.post("/upload")
    async def upload(request: Request, invoice: UploadFile = File(...), photos: list[UploadFile] | None = File(None),
                     uploader: str = Form("Demo user")):
        photos = [p for p in (photos or []) if p.filename]
        inv_suffix = Path(invoice.filename or "").suffix.lower()
        if inv_suffix not in IMAGE_SUFFIXES | {".pdf"}:
            return upload_form(request, error="The invoice must be a PDF or an image (JPG, PNG, HEIC...).")
        if any(Path(p.filename).suffix.lower() not in IMAGE_SUFFIXES for p in photos):
            return upload_form(request, error="Photos must be images (JPG, PNG, HEIC...).")
        if len(photos) > MAX_PHOTOS:
            return upload_form(request, error=f"At most {MAX_PHOTOS} photos per upload.")
        blobs = [(f"invoice{inv_suffix}", await invoice.read())]
        blobs += [(f"photos/{i:02d}_{safe_name(p.filename)}", await p.read()) for i, p in enumerate(photos, 1)]
        if any(len(b) > MAX_UPLOAD_BYTES for _, b in blobs) or not blobs[0][1]:
            return upload_form(request, error="Empty file, or a file over 40 MB.")

        def save(prefix: str):
            keys = [storage.put_bytes(f"{prefix}/{name}", data) for name, data in blobs]
            return keys[0], keys[1:]

        return go(request, f"/upload/{_start_job(uploader, save)}")

    @app.post("/upload/sample")
    def upload_sample(request: Request):
        sample = jobs.sample_files(storage)
        if sample is None:
            return upload_form(request, error="No sample: run python tools/seed_demo.py first.")
        inv_path, photo_paths = sample

        def save(prefix: str):
            inv = storage.put_file(f"{prefix}/invoice{inv_path.suffix}", inv_path)
            return inv, [storage.put_file(f"{prefix}/photos/{p.name}", p) for p in photo_paths]

        return go(request, f"/upload/{_start_job('Demo user', save)}")

    def _job_ctx(job_id: int) -> dict:
        with sessions() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise HTTPException(404, "upload not found")
            clusters = [db.get(Cluster, cid) for cid in job.cluster_ids]
            for c in clusters:
                _ = c.photos, c.invoice and c.invoice.vehicle
            inv = db.get(Invoice, job.invoice_id) if job.invoice_id else None
            return {"job": job, "clusters": clusters, "inv": inv}

    @app.get("/upload/{job_id}", response_class=HTMLResponse)
    def upload_status_page(request: Request, job_id: int):
        return render(request, "upload_job.html", active="upload", **_job_ctx(job_id))

    @app.get("/upload/{job_id}/status", response_class=HTMLResponse)
    def upload_status(request: Request, job_id: int):
        return render(request, "_job_status.html", active="upload", **_job_ctx(job_id))

    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    """``app.main:app`` for uvicorn, created on first use (importing create_app in tests
    touches no data folder)."""
    global _app
    if name == "app":
        if _app is None:
            logging.basicConfig(level=logging.INFO)
            _app = create_app()
        return _app
    raise AttributeError(name)
