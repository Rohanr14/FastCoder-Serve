"""Best-effort capture of vLLM speculative-decoding acceptance from its Prometheus /metrics.

Speculative decoding's headline quality-of-speculation metric is the **acceptance rate** — how many
drafted tokens the target model verifies and keeps. vLLM exposes spec-decode counters on its metrics
endpoint, but the exact names have changed across versions, so this parses defensively: it sums any
counter whose name contains ``spec_decode`` plus ``accepted``/``draft`` and ``token``, and returns
accepted/draft, or ``None`` when those metrics are absent (a build that names them differently, or a
non-speculative server). It never fabricates a number — null means "not derivable here, read the raw
lines."
"""

from __future__ import annotations

import re

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eEnaN]+)\s*$")


def spec_decode_lines(metrics_text: str) -> list[str]:
    """Return the raw (non-comment) ``spec_decode`` sample lines for transparent inspection."""

    lines: list[str] = []
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "spec_decode" in line.lower():
            lines.append(line)
    return lines


def _sum_matching(metrics_text: str, *, must_contain: tuple[str, ...]) -> float | None:
    total = 0.0
    found = False
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name").lower()
        if all(token in name for token in must_contain):
            try:
                total += float(match.group("value"))
            except ValueError:
                continue
            found = True
    return total if found else None


def spec_decode_acceptance(metrics_text: str) -> float | None:
    """Return drafted-token acceptance (accepted / draft) from vLLM /metrics text, or ``None``."""

    accepted = _sum_matching(metrics_text, must_contain=("spec_decode", "accepted", "token"))
    draft = _sum_matching(metrics_text, must_contain=("spec_decode", "draft", "token"))
    if accepted is None or draft is None or draft <= 0:
        return None
    return accepted / draft
