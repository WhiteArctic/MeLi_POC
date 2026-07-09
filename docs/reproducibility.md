# Reproducibility

This repository is designed to be shared without large or sensitive artifacts. A
fresh clone contains source code, tests, documentation, final lightweight JSON
reports, and manifests. It does not include the original challenge CSV, derived CSV splits,
trained model binaries, image caches, or generated documents.

## Clean-clone checklist

After cloning, the tree should contain:

```text
README.md
Makefile
pyproject.toml
docs/
scripts/
src/
tests/
outputs/**/*.json
```

The following files are local inputs or generated artifacts and should not be
required in Git:

```text
anexo_1_dataset.csv
Reto*.pdf
outputs/datasets/*.csv
outputs/golden_set/*.csv
outputs/models/*.pt
outputs/models/*.npz
outputs/reports/*.details.csv
outputs/documents/
outputs/image_cache/
outputs/model_cache/
```

## Environment setup

Create the base environment and install the package:

```bash
make setup
```

Run unit tests:

```bash
make test
```

These tests do not require the original dataset.

## Required input data

To rebuild datasets and train models, place the source CSV in the project root:

```text
anexo_1_dataset.csv
```

Expected columns:

```text
picture_url,infraction_detected,labels_detected,ocr_text
```

The source CSV is ignored by Git and must be provided through an approved external
channel, such as a private drive, a data catalog, or the original challenge
package.

## Rebuild pipeline

Build the golden set and leakage-safe train/validation/test splits:

```bash
make datasets
```

Expected generated files:

```text
outputs/golden_set/golden_set_v1.csv
outputs/golden_set/golden_manifest_v1.json
outputs/golden_set/golden_strategy_v1.md
outputs/datasets/train.csv
outputs/datasets/validation.csv
outputs/datasets/test_internal.csv
outputs/datasets/dataset_manifest.json
```

Train the baseline text model:

```bash
make train-text
```

Expected generated files:

```text
outputs/models/text_nb_model_v1.npz
outputs/models/text_nb_model_v1.json
```

Run offline baseline evaluations:

```bash
make baselines
```

Run a small end-to-end smoke test from `picture_url`:

```bash
make smoke
```

## Candidate pipeline

Install optional OCR and visual dependencies:

```bash
make setup-candidate
```

Train the MobileNetV3 visual fusion model:

```bash
make train-visual-pretrained
```

Run the measured candidate validation:

```bash
make validate-candidate
```

Run a larger validation sample:

```bash
make validate-candidate-large
```

## Determinism and expected variation

Dataset partitioning uses stable hashes over `leakage_group_id`, so the selected
groups should be stable as long as `anexo_1_dataset.csv` and the code do not
change.

End-to-end metrics may vary slightly because they depend on remote image
availability, local cache state, OCR library versions, pretrained weights, CPU/GPU
hardware, and network latency.

## Artifact policy

Small JSON reports and manifests remain versioned to make the measured state of
the solution inspectable without downloading heavy data. Regenerated CSV details,
model binaries, image caches, and rendered documents are intentionally local.

Use:

```bash
make clean-generated
make clean-local
```

before creating a clean ZIP or handing the project to another reviewer.
