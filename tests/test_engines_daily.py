"""Daily tick: key-from-env loading, state store without a repo, report rendering."""

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.clients.kalshi_client import _pem_from_env
from src.engines.daily import DailyResult, StateStore, live_credentials, render


@pytest.fixture
def pem():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


def test_private_key_from_env_in_every_shape(monkeypatch, pem):
    for value in (pem, pem.replace("\n", "\\n"), base64.b64encode(pem.encode()).decode()):
        monkeypatch.setenv("KALSHI_PRIVATE_KEY", value)
        serialization.load_pem_private_key(_pem_from_env(), password=None)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY")
    assert _pem_from_env() is None


def test_live_credentials_need_both_key_id_and_key(monkeypatch, tmp_path, pem):
    monkeypatch.delenv("KALSHI_API_KEY", raising=False)
    monkeypatch.setenv("KALSHI_PRIVATE_KEY", pem)
    assert not live_credentials()
    monkeypatch.setenv("KALSHI_API_KEY", "id")
    assert live_credentials()


def test_state_store_without_repo_says_record_wont_persist():
    s = StateStore(None, log=lambda *_: None)
    s.restore()
    s.save("x")  # no-op, no crash
    assert not s.ok and "will not persist" in s.note


def test_report_renders_trust_and_live_sections():
    r = DailyResult(date="2026-10-02", state="persisted in me/state",
                    promotions={"games": {"settled": 3, "pnl": 1.2, "z": 0.0, "weight": 0.0,
                                          "verdict": "not enough settled shadow orders"}},
                    live_note="Engines with earned trust: none.")
    md = render(r)
    assert "| games | 3 |" in md and "Engines with earned trust: none." in md
