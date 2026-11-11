import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import Settings, get_settings
from src.common.exceptions import LLMGatewayError
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GatewayMetrics:
    in_flight_ingest: int = 0
    interactive_active: int = 0
    errors: int = 0
    total_ingest_requests: int = 0
    total_answer_requests: int = 0


class LLMPriorityGateway:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Ingestion client (throughput optimized, stream=False)
        self.ingest_client = AsyncOpenAI(
            base_url=self.settings.llm_ingest.base_url,
            api_key="EMPTY",
            timeout=httpx.Timeout(self.settings.llm_ingest.timeout),
            max_retries=0,  # Handled via tenacity
        )
        self.ingest_semaphore = asyncio.Semaphore(self.settings.llm_ingest.max_concurrency)

        # Answering client (latency optimized, stream=True)
        self.answer_client = AsyncOpenAI(
            base_url=self.settings.llm_answer.base_url,
            api_key="EMPTY",
            timeout=httpx.Timeout(self.settings.llm_answer.timeout),
            max_retries=0,
        )

        self.metrics = GatewayMetrics()
        self._lock = asyncio.Lock()

    async def _update_metrics(self, ingest_delta: int = 0, interactive_delta: int = 0, error_delta: int = 0):
        async with self._lock:
            self.metrics.in_flight_ingest += ingest_delta
            self.metrics.interactive_active += interactive_delta
            self.metrics.errors += error_delta

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True
    )
    async def _extract_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Backpressure hook: In a full implementation, this semaphore would be dynamic.
        # For this scaffold, we rely on the fixed semaphore and track interactive_active for metrics.
        async with self.ingest_semaphore:
            self.metrics.total_ingest_requests += 1
            await self._update_metrics(ingest_delta=1)
            try:
                response = await self.ingest_client.chat.completions.create(
                    model=self.settings.llm_ingest.model,
                    messages=payload["messages"],
                    stream=False,  # Strictly non-streaming for ingestion
                    temperature=payload.get("temperature", 0.1),
                    max_tokens=payload.get("max_tokens", 4096),
                )
                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.model_dump() if response.usage else None
                }
            except Exception as e:
                await self._update_metrics(error_delta=1)
                logger.error("Ingestion LLM call failed", error=str(e))
                raise LLMGatewayError(f"Ingestion failed: {e}") from e
            finally:
                await self._update_metrics(ingest_delta=-1)

    async def extract_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities/relations in batch using concurrent requests (no manual prompt batching)."""
        tasks = [self._extract_single(p) for p in payloads]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def answer_stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream final answer with low latency."""
        await self._update_metrics(interactive_delta=1)
        self.metrics.total_answer_requests += 1
        try:
            stream = await self.answer_client.chat.completions.create(
                model=self.settings.llm_answer.model,
                messages=payload["messages"],
                stream=True,  # Strictly streaming for final answer
                temperature=payload.get("temperature", 0.3),
                max_tokens=payload.get("max_tokens", 2048),
                tools=payload.get("tools"),
                tool_choice=payload.get("tool_choice"),
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            await self._update_metrics(error_delta=1)
            logger.error("Answer streaming failed", error=str(e))
            raise LLMGatewayError(f"Answer streaming failed: {e}") from e
        finally:
            await self._update_metrics(interactive_delta=-1)

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "in_flight_ingest": self.metrics.in_flight_ingest,
            "interactive_active": self.metrics.interactive_active,
            "errors": self.metrics.errors,
            "total_ingest_requests": self.metrics.total_ingest_requests,
            "total_answer_requests": self.metrics.total_answer_requests,
        }

    async def close(self):
        await self.ingest_client.close()
        await self.answer_client.close()