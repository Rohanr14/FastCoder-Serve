PYTHON ?= $(shell if command -v python3.11 >/dev/null 2>&1; then command -v python3.11; elif [ -x /opt/homebrew/bin/python3.11 ]; then echo /opt/homebrew/bin/python3.11; elif [ -x /usr/local/bin/python3.11 ]; then echo /usr/local/bin/python3.11; elif [ -x /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 ]; then echo /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11; else command -v python3; fi)

.PHONY: install lint typecheck test ci fake-server gateway bench-smoke bench-smoke-streaming plot-smoke validate-smoke-config validate-streaming-config validate-baseline-config check-fake-endpoint validate-smoke-results validate-streaming-results manifest-baseline baseline-dry-run

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy fastcoder

test:
	$(PYTHON) -m pytest

ci: lint typecheck test

fake-server:
	$(PYTHON) scripts/fake_openai_server.py --host 127.0.0.1 --port 9000

gateway:
	$(PYTHON) scripts/run_gateway.py --host 127.0.0.1 --port 8000

bench-smoke:
	$(PYTHON) scripts/run_bench.py --config configs/local_smoke.yaml --start-fake-server

bench-smoke-streaming:
	$(PYTHON) scripts/run_bench.py --config configs/local_smoke_streaming.yaml --start-fake-server

plot-smoke:
	$(PYTHON) scripts/generate_pareto.py results/local_smoke.json --output results/pareto.png

validate-smoke-config:
	$(PYTHON) scripts/validate_config.py --config configs/local_smoke.yaml

validate-streaming-config:
	$(PYTHON) scripts/validate_config.py --config configs/local_smoke_streaming.yaml

validate-baseline-config:
	$(PYTHON) scripts/validate_config.py --config configs/baseline_fp16.yaml

check-fake-endpoint:
	$(PYTHON) scripts/check_endpoint.py --base-url http://localhost:9000/v1 --model fake-coder --timeout-seconds 5

validate-smoke-results:
	$(PYTHON) scripts/validate_results.py results/local_smoke.json

validate-streaming-results:
	$(PYTHON) scripts/validate_results.py results/local_smoke_streaming.json

manifest-baseline:
	$(PYTHON) scripts/create_run_manifest.py --config configs/baseline_fp16.yaml --notes "H100 FP16 baseline pre-flight manifest"

baseline-dry-run:
	$(PYTHON) scripts/run_h100_baseline.py --config configs/baseline_fp16.yaml --base-url http://localhost:8000/v1
