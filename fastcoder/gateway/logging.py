"""Structured JSON logging helpers."""

from __future__ import annotations

import json
import logging as stdlib_logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(stdlib_logging.Formatter):
    """Minimal JSON log formatter with stable fields."""

    def format(self, record: stdlib_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("fastcoder_"):
                payload[key.removeprefix("fastcoder_")] = value
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str) -> None:
    """Configure package logging for the gateway."""

    logger = stdlib_logging.getLogger("fastcoder")
    logger.handlers.clear()
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
