from __future__ import annotations

import json
import re
from pathlib import Path

DASHBOARD = Path("observability/grafana/fastcoder-dashboard.json")
METRICS_SRC = Path("fastcoder/gateway/metrics.py")


def _declared_metric_names() -> set[str]:
    source = METRICS_SRC.read_text(encoding="utf-8")
    return set(re.findall(r'"(fastcoder_gateway_[a-z_]+)"', source))


def test_dashboard_is_valid_json_with_panels() -> None:
    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert data["panels"], "dashboard has no panels"
    assert data["title"]


def test_dashboard_only_references_real_gateway_metrics() -> None:
    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    exprs = [target["expr"] for panel in data["panels"] for target in panel.get("targets", [])]
    assert exprs, "dashboard has no metric expressions"
    declared = _declared_metric_names()
    assert declared
    referenced = set(re.findall(r"fastcoder_gateway_[a-z_]+", " ".join(exprs)))
    assert referenced
    for metric in referenced:
        base = metric
        for suffix in ("_bucket", "_count", "_sum"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        assert base in declared, f"dashboard references undefined metric {metric!r} (base {base!r})"


def test_demo_load_module_imports_and_exposes_main() -> None:
    import scripts.demo_load as demo

    assert callable(demo.main)
