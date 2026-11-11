import pytest
from pydantic import ValidationError
from config.settings import Settings, LLMIngestSettings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_INGEST_BASE_URL", "http://test:8000/v1")
    monkeypatch.setenv("LLM_INGEST_MODEL", "test-model")
    monkeypatch.setenv("QDRANT_URL", "http://test-qdrant:6333")

    settings = Settings()
    assert settings.llm_ingest.base_url == "http://test:8000/v1"
    assert settings.llm_ingest.model == "test-model"
    assert settings.qdrant.url == "http://test-qdrant:6333"


def test_frozen_settings_reject_mutation():
    settings = LLMIngestSettings()
    with pytest.raises(ValidationError):
        settings.base_url = "http://new-url/v1"