import pytest
from unittest.mock import AsyncMock, MagicMock
from src.llm.gateway import LLMPriorityGateway
from config.settings import Settings


@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.llm_ingest.base_url = "http://mock-ingest:8000/v1"
    settings.llm_answer.base_url = "http://mock-answer:8001/v1"
    return settings


@pytest.mark.asyncio
async def test_extract_batch_uses_stream_false(mock_settings):
    gateway = LLMPriorityGateway(settings=mock_settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="test content"))]
    mock_response.usage = None
    gateway.ingest_client.chat.completions.create = AsyncMock(return_value=mock_response)

    payloads = [{"messages": [{"role": "user", "content": "test"}]}]
    await gateway.extract_batch(payloads)

    gateway.ingest_client.chat.completions.create.assert_called_once()
    call_kwargs = gateway.ingest_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is False


@pytest.mark.asyncio
async def test_answer_stream_uses_stream_true(mock_settings):
    gateway = LLMPriorityGateway(settings=mock_settings)

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="test "))]

    async def mock_stream(*args, **kwargs):
        yield mock_chunk

    gateway.answer_client.chat.completions.create = mock_stream

    payload = {"messages": [{"role": "user", "content": "test"}]}
    result = []
    async for chunk in gateway.answer_stream(payload):
        result.append(chunk)

    assert result == ["test "]
