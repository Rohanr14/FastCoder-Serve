"""Best-effort capture of vLLM speculative-decoding acceptance from its Prometheus /metrics.

Speculative decoding's headline quality-of-speculation metric is the **acceptance rate** — the
fraction of drafted tokens the target model verifies and keeps:
``spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total``.

vLLM exposes those as ``*_total`` counters, but it also exposes ``*_created`` series (Prometheus
metric-creation **timestamps**, ~1.78e9 epoch seconds) and ``*_per_pos`` breakdowns. We match the
``*_total`` counters **exactly** (by suffix) so those are never summed in — an earlier version that
matched on substrings summed the ``_created`` epochs and produced a nonsensical ratio > 1. Returns
``None`` when the counters are absent or no draft tokens were recorded (e.g. capturing against a
freshly-restarted server), rather than fabricating a number.
"""

from __future__ import annotations

import re

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")

_ACCEPTED_SUFFIX = "spec_decode_num_accepted_tokens_total"
_DRAFT_SUFFIX = "spec_decode_num_draft_tokens_total"


def spec_decode_lines(metrics_text: str) -> list[str]:
    """Return the raw (non-comment) ``spec_decode`` sample lines for transparent inspection."""

    lines: list[str] = []
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "spec_decode" in line.lower():
            lines.append(line)
    return lines


def _sum_named(metrics_text: str, suffix: str) -> float | None:
    """Sum the values of samples whose metric name ends with ``suffix`` (None if none match)."""

    total = 0.0
    found = False
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None or not match.group("name").endswith(suffix):
            continue
        try:
            total += float(match.group("value"))
        except ValueError:
            continue
        found = True
    return total if found else None


def spec_decode_acceptance(metrics_text: str) -> float | None:
    """Return accepted/draft token acceptance from vLLM /metrics text, or ``None``.

    Uses the ``*_total`` counters only; ignores ``*_created`` timestamps and ``*_per_pos`` series.
    """

    accepted = _sum_named(metrics_text, _ACCEPTED_SUFFIX)
    draft = _sum_named(metrics_text, _DRAFT_SUFFIX)
    if accepted is None or draft is None or draft <= 0:
        return None
    return accepted / draft
