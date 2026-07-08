PYTHON ?= python3
OCR_PYTHON ?= .venv-ocr/bin/python

.PHONY: setup setup-candidate build-datasets train-model test smoke validate-candidate

setup:
	$(PYTHON) -m pip install -e .

setup-candidate:
	$(PYTHON) -m venv .venv-ocr
	$(OCR_PYTHON) -m pip install -U pip
	$(OCR_PYTHON) -m pip install -e ".[candidate]"

build-datasets:
	PYTHONPATH=src $(PYTHON) scripts/build_datasets.py

train-model:
	PYTHONPATH=src $(PYTHON) scripts/train_model.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) scripts/evaluate_end_to_end.py --dataset-csv outputs/datasets/validation.csv --rows-per-class 2 --output-json outputs/reports/smoke_latest.json --progress-every 1

validate-candidate:
	PYTHONPATH=src $(OCR_PYTHON) scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --dedupe-group-key leakage_group_id --output-json outputs/reports/candidate_validation_latest.json --progress-every 10
