PYTHON ?= /Users/O001224/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
OCR_PYTHON ?= .venv-ocr/bin/python

.PHONY: setup-candidate test smoke validate-candidate

setup-candidate:
	python -m venv .venv-ocr
	$(OCR_PYTHON) -m pip install -U pip
	$(OCR_PYTHON) -m pip install -e ".[candidate]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) scripts/evaluate_end_to_end.py --dataset-csv outputs/datasets/validation.csv --rows-per-class 2 --output-json outputs/reports/smoke_latest.json --progress-every 1

validate-candidate:
	PYTHONPATH=src $(OCR_PYTHON) scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --dedupe-group-key leakage_group_id --output-json outputs/reports/candidate_validation_latest.json --progress-every 10
