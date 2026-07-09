PYTHON ?= python3
VENV ?= .venv
CANDIDATE_VENV ?= .venv-ocr
RUN_PYTHON ?= $(VENV)/bin/python
OCR_PYTHON ?= $(CANDIDATE_VENV)/bin/python
PYTHONPATH ?= src

.PHONY: help setup setup-candidate test golden build-datasets datasets train-model train-text baselines smoke train-visual train-visual-pretrained validate-candidate validate-candidate-large diagnose diagnose-candidate diagnose-debug clean-generated clean-local

help:
	@printf "Available targets:\n"
	@printf "  setup                    Create base virtualenv and install package\n"
	@printf "  setup-candidate          Create OCR/visual virtualenv with optional deps\n"
	@printf "  test                     Run unit tests\n"
	@printf "  golden                   Build leakage-safe golden set from source CSV\n"
	@printf "  datasets                 Build train/validation/test_internal splits\n"
	@printf "  train-text               Train text Naive Bayes model\n"
	@printf "  baselines                Evaluate offline text/regex baselines\n"
	@printf "  smoke                    Run a 4-row end-to-end smoke test\n"
	@printf "  train-visual             Train lightweight visual feature model\n"
	@printf "  train-visual-pretrained  Train MobileNetV3 visual fusion model\n"
	@printf "  validate-candidate       Run candidate validation on 100 balanced rows\n"
	@printf "  validate-candidate-large Run candidate validation on 500 balanced rows\n"
	@printf "  diagnose                 Diagnose one URL with concise public output, pass URL=\"https://...\"\n"
	@printf "  diagnose-candidate       Diagnose one URL with OCR/visual candidate deps\n"
	@printf "  diagnose-debug           Diagnose one URL with full internal metadata\n"
	@printf "  clean-generated          Remove generated datasets/models/details/docs\n"
	@printf "  clean-local              Remove virtualenvs and Python/macOS caches\n"

setup:
	$(PYTHON) -m venv $(VENV)
	$(RUN_PYTHON) -m pip install -U pip
	$(RUN_PYTHON) -m pip install -e .

setup-candidate:
	$(PYTHON) -m venv $(CANDIDATE_VENV)
	$(OCR_PYTHON) -m pip install -U pip
	$(OCR_PYTHON) -m pip install -e ".[candidate]"

test:
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) -m unittest discover -s tests -v

golden:
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/generate_golden_set.py

build-datasets: datasets

datasets: golden
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/build_datasets.py

train-model: train-text

train-text: datasets
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/train_model.py

baselines: train-text
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/evaluate_model.py --dataset-csv outputs/datasets/validation.csv --output-json outputs/reports/validation_text_model.json
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/compare_baselines.py outputs/datasets/validation.csv outputs/reports/validation_baseline_comparison.json

smoke: train-text
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/evaluate_end_to_end.py --dataset-csv outputs/datasets/validation.csv --rows-per-class 2 --dedupe-group-key leakage_group_id --output-json outputs/reports/end_to_end_validation_smoke.json --progress-every 1

train-visual: datasets
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/train_visual_model.py --rows-per-class 500 --validation-rows-per-class 200 --output-json outputs/reports/visual_model_validation_500.json

train-visual-pretrained: datasets
	PYTHONPATH=$(PYTHONPATH) $(OCR_PYTHON) scripts/train_pretrained_visual_model.py --rows-per-class 500 --validation-rows-per-class 200 --output-json outputs/reports/pretrained_visual_validation_500.json

validate-candidate:
	PYTHONPATH=$(PYTHONPATH) $(OCR_PYTHON) scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --dedupe-group-key leakage_group_id --output-json outputs/reports/candidate_validation_100_unique_latest.json --progress-every 10

validate-candidate-large:
	PYTHONPATH=$(PYTHONPATH) $(OCR_PYTHON) scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 250 --dedupe-group-key leakage_group_id --output-json outputs/reports/candidate_validation_500_unique_latest.json --progress-every 25

diagnose:
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/diagnose_url.py "$(URL)"

diagnose-candidate:
	PYTHONPATH=$(PYTHONPATH) $(OCR_PYTHON) scripts/diagnose_url.py "$(URL)" --candidate

diagnose-debug:
	PYTHONPATH=$(PYTHONPATH) $(RUN_PYTHON) scripts/diagnose_url.py "$(URL)" --verbose

clean-generated:
	rm -rf outputs/datasets/*.csv outputs/golden_set/*.csv outputs/models/*.npz outputs/models/*.pt outputs/reports/*.details.csv outputs/documents reporte_tecnico_solucion_poc.pages

clean-local:
	rm -rf $(VENV) $(CANDIDATE_VENV) .pytest_cache .ruff_cache .mypy_cache
	find . -path ./.git -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -path ./.git -prune -o -name .DS_Store -type f -exec rm -f {} +
