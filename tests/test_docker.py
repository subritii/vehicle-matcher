"""Docker setup: the container's start-up step and the image / compose / Render files."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import docker_start as ds  # noqa: E402


def read(name: str) -> str:
    return (ROOT / name).read_text()


# --- Start-up: restore the scale demo, or build it once ------------------------------------


class Calls:
    def __init__(self, seed_rc=0):
        self.log, self.seed_rc = [], seed_rc

    def restore(self, data):
        self.log.append(("restore", Path(data)))

    def seed(self, argv):
        self.log.append(("seed", argv))
        (Path(argv[argv.index("--data") + 1]) / "snapshot.db").write_bytes(b"db")
        return self.seed_rc

    def fetch(self, contact):
        self.log.append(("fetch", contact))
        return 0


def test_existing_snapshot_is_restored(tmp_path):
    (tmp_path / "snapshot.db").write_bytes(b"db")
    c = Calls()
    assert ds.prepare(tmp_path, tmp_path / "photos", 200, "", restore=c.restore, seed=c.seed, fetch=c.fetch,
                      have_exteriors=lambda: True) == "restored"
    assert c.log == [("restore", tmp_path)]


def test_first_start_builds_the_scale_demo(tmp_path):
    c = Calls()
    assert ds.prepare(tmp_path, tmp_path / "photos", 300, "me@example.com", restore=c.restore, seed=c.seed,
                      fetch=c.fetch, have_exteriors=lambda: True) == "built"
    (kind, argv), = c.log
    assert kind == "seed"
    assert argv[argv.index("--count") + 1] == "300"
    assert argv[argv.index("--data") + 1] == str(tmp_path)
    assert argv[argv.index("--photos") + 1] == str(tmp_path / "photos")


def test_exteriors_are_downloaded_first_when_missing(tmp_path):
    c = Calls()
    ds.prepare(tmp_path, tmp_path / "p", 200, "me@example.com", restore=c.restore, seed=c.seed, fetch=c.fetch,
               have_exteriors=lambda: False)
    assert [x[0] for x in c.log] == ["fetch", "seed"]
    assert c.log[0] == ("fetch", "me@example.com")


def test_missing_contact_explains_what_to_set(tmp_path):
    c = Calls()
    with pytest.raises(SystemExit, match="WIKIMEDIA_CONTACT"):
        ds.prepare(tmp_path, tmp_path / "p", 200, "", restore=c.restore, seed=c.seed, fetch=c.fetch,
                   have_exteriors=lambda: False)
    assert c.log == []


def test_failed_build_stops_the_container(tmp_path):
    c = Calls(seed_rc=1)
    with pytest.raises(SystemExit, match="scale demo"):
        ds.prepare(tmp_path, tmp_path / "p", 200, "x", restore=c.restore, seed=c.seed, fetch=c.fetch,
                   have_exteriors=lambda: True)


def test_server_command_uses_the_scale_demo(tmp_path):
    argv, env = ds.server(tmp_path, port=8000, base_env={"PATH": "/usr/bin"})
    assert argv[:2] == ["uvicorn", "app.main:app"] and "0.0.0.0" in argv and "8000" in argv
    assert env["DATA_DIR"] == str(tmp_path)
    assert env["DATABASE_URL"] == f"sqlite:///{tmp_path / 'app.db'}"
    assert "ALLOW_PUBLIC_CLIENTS" not in env  # Docker's port forwarding arrives from a private address


# --- The files -----------------------------------------------------------------------------


def test_full_image():
    d = read("Dockerfile")
    assert re.search(r"^FROM python:3\.12(\.\d+)?-slim", d, re.M)
    assert "requirements-ocr.txt" in d and "requirements-ocr-excludes.txt" in d
    assert re.search(r"^USER (?!root)\w+", d, re.M)
    assert "HEALTHCHECK" in d and "/healthz" in d


def test_showcase_image_is_slim():
    d = read("Dockerfile.showcase")
    assert re.search(r"^FROM python:3\.12(\.\d+)?-slim", d, re.M)
    assert "requirements-showcase.txt" in d and "requirements-ocr" not in d
    assert "DEMO_MODE=showcase" in d
    assert re.search(r"^USER (?!root)\w+", d, re.M)
    assert "HEALTHCHECK" in d and "/healthz" in d
    assert "--proxy-headers" in d  # rate limiting sees the visitor, not Render's proxy


def test_showcase_requirements_leave_out_ocr_and_ai():
    reqs = read("requirements-showcase.txt").lower()
    for heavy in ("paddle", "fast-alpr", "opencv", "onnx", "pillow", "pdfplumber", "pypdfium2", "torch"):
        assert heavy not in reqs


def test_compose_runs_the_app_and_ollama_with_a_lite_profile():
    c = read("docker-compose.yml")
    assert "ollama/ollama" in c and "qwen3.5:4b-q4_K_M" in c
    assert re.search(r"profiles:\s*\[\s*lite\s*\]", c) and re.search(r"profiles:\s*\[\s*full\s*\]", c)
    assert "127.0.0.1:8000:8000" in c  # never on the LAN or internet by accident
    for volume in ("app-data", "models", "exteriors", "ollama"):
        assert re.search(rf"^  {volume}:", c, re.M), volume
    assert "service_healthy" in c


def test_compose_mac_profile_uses_ollama_on_the_host():
    """Docker on a Mac has no GPU: the mac profile runs the app in Docker and the model natively."""
    c = read("docker-compose.yml")
    assert re.search(r"profiles:\s*\[\s*mac\s*\]", c)
    assert "http://host.docker.internal:11434" in c and "host-gateway" in c


def test_render_blueprint_deploys_the_showcase_image():
    r = read("render.yaml")
    assert "dockerfilePath: ./Dockerfile.showcase" in r
    assert "plan: free" in r and "healthCheckPath: /healthz" in r
    assert "DEMO_MODE" in r and "showcase" in r


def test_dockerignore_keeps_private_files_out_of_images():
    ignore = read(".dockerignore").split()
    for pattern in ("data", ".env", "tests/fixtures/private", "claude.md", "CLAUDE.md", ".venv", ".git"):
        assert pattern in ignore, pattern
