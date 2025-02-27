from __future__ import annotations

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(path)s %(method)s %(status_code)s %(duration_ms)s"
    )
    handler.setFormatter(formatter)

    root.addHandler(handler)
    root.setLevel(level.upper())


class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        for key in ["request_id", "path", "method", "status_code", "duration_ms"]:
            extra.setdefault(key, None)
        return msg, kwargs
