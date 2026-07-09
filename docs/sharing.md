# Sharing checklist

Use this checklist before sharing the project with another data scientist or
reviewer.

## Include

```text
README.md
Makefile
pyproject.toml
.gitattributes
.gitignore
docs/
scripts/
src/
tests/
outputs/**/*.json
```

Keep only final/relevant report JSON files in `outputs/reports/`; intermediate
smoke tests and experiment variants can be regenerated and should not be shared.

## Share separately when allowed

```text
anexo_1_dataset.csv
Reto*.pdf
```

These files are inputs, not source code. They may be large or sensitive, so they
should travel through an approved data channel rather than normal Git history.

## Do not include

```text
.DS_Store
__pycache__/
.venv/
.venv-ocr/
outputs/datasets/*.csv
outputs/golden_set/*.csv
outputs/models/*.pt
outputs/models/*.npz
outputs/reports/*.details.csv
outputs/documents/
outputs/image_cache/
outputs/model_cache/
```

## Verify before sharing

```bash
make setup
make test
make clean-generated
make clean-local
git status --short
```

If the recipient has the source CSV, they can rebuild the generated artifacts with:

```bash
make datasets
make train-text
make baselines
```
