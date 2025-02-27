from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class TraceClient:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.langfuse_enabled
        self._client = None
        if not self.enabled:
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning("Langfuse tracing enabled but keys are missing; tracing disabled.")
            self.enabled = False
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                environment=settings.langfuse_environment,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            self.enabled = False

    def capture(
        self,
        *,
        name: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._client is None:
            return

        try:
            trace = self._client.trace(name=name, input=input_payload, metadata=metadata or {})
            if output_payload is not None:
                trace.update(output=output_payload)
            self._client.flush()
        except Exception as exc:  # pragma: no cover
            logger.warning("Langfuse capture failed: %s", exc)


@lru_cache(maxsize=1)
def get_trace_client() -> TraceClient:
    return TraceClient(get_settings())
