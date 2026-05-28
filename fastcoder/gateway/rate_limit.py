"""Tiny in-memory rate limiter for local gateway demos.

This limiter is intentionally non-distributed. It is suitable for local smoke tests and demos, not
for multi-process or multi-instance production enforcement.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    reset_seconds: float


class InMemoryRateLimiter:
    """Sliding-window rate limiter keyed by token or client IP."""

    def __init__(
        self,
        *,
        limit_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit_per_minute < 1:
            msg = "limit_per_minute must be >= 1"
            raise ValueError(msg)
        self._limit = limit_per_minute
        self._clock = clock
        self._window_seconds = 60.0
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> RateLimitDecision:
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window_seconds:
            hits.popleft()

        if len(hits) >= self._limit:
            reset_seconds = max(0.0, self._window_seconds - (now - hits[0]))
            return RateLimitDecision(allowed=False, remaining=0, reset_seconds=reset_seconds)

        hits.append(now)
        remaining = max(0, self._limit - len(hits))
        reset_seconds = self._window_seconds if not hits else self._window_seconds - (now - hits[0])
        return RateLimitDecision(allowed=True, remaining=remaining, reset_seconds=reset_seconds)
