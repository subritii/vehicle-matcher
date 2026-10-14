import pytest

from app.extract import vin


@pytest.fixture(autouse=True)
def _isolate_vpic(request, monkeypatch):
    vin.clear_vpic_cache()
    monkeypatch.delenv("VIN_MARKET", raising=False)  # tests start from the default
    if "live" not in request.keywords:
        # Any test that forgets to inject a mock client fails fast instead of hitting NHTSA.
        monkeypatch.setenv("VPIC_BASE_URL", "http://vpic.test.invalid/api/vehicles")
    yield
    vin.clear_vpic_cache()
