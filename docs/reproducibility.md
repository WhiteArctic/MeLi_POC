# Reproducibility

This repository intentionally does not include the original dataset, derived CSV splits, trained model binaries, or generated documents. Those files are local artifacts and may be large or sensitive.

## What works after cloning

After cloning the repository, users can run the unit tests and inspect the code, docs, metrics JSON files, and dataset/model manifests without downloading large artifacts.

```bash
python3 -m pip install -e .
make test
```

## Required local input data

To rebuild datasets and train models, place the source CSV at the project root:

```text
anexo_1_dataset.csv
```

The expected columns are:

```text
picture_url,infraction_detected,labels_detected,ocr_text
```

The source CSV is ignored by Git and must be provided through an approved external channel, such as a private drive, a data catalog, or the original challenge package.

## Rebuild generated artifacts

Create the leakage-safe dataset splits:

```bash
make build-datasets
```

Train the baseline text model:

```bash
make train-model
```

Run a quick validation smoke test:

```bash
make smoke
```

Run the candidate validation when optional OCR and visual dependencies are available:

```bash
make setup-candidate
make validate-candidate
```

## Artifact policy

The following paths are intentionally ignored:

```text
anexo_1_dataset.csv
outputs/datasets/*.csv
outputs/golden_set/*.csv
outputs/models/*.pt
outputs/models/*.npz
outputs/reports/*.details.csv
outputs/documents/
```

Small JSON reports and manifests remain versioned so users can understand the measured state of the solution without downloading heavy data.
