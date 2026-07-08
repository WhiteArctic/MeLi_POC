# Meli Image Moderation POC

POC para el reto tecnico de moderacion de imagenes. El objetivo es detectar imagenes con textos o badges no permitidos y devolver:

```python
{"has_infraction": bool, "evidence": str}
```

## Flujo de datasets

1. `golden_set_v1`: examen final. No se usa para entrenar, ajustar umbrales ni seleccionar modelos.
2. `train`: entrenamiento del modelo textual usando OCR legacy como etiqueta debil.
3. `validation`: seleccion de umbrales y comparacion de variantes.
4. `test_internal`: evaluacion interna antes de tocar el golden.

La particion se hace por `leakage_group_id`, que corresponde al `item_id` parseado desde `picture_url`. Asi se evita que imagenes del mismo item caigan en conjuntos distintos.

## Arquitectura

```text
picture_url
  -> descarga con urllib, SSL verify=False y cache local
  -> OCR local reemplazable; candidato usa EasyOCR si esta instalado
  -> normalizacion, correcciones OCR y sinonimos
  -> fuzzy matching de terminos de politica
  -> clasificador semantico textual por chunks
  -> detector visual heuristico de overlays/badges
  -> clasificador visual MobileNetV3 opcional
  -> decision hibrida con soporte y abstencion
  -> has_infraction + evidence
```

La POC evita dependencias pesadas para poder correr en red corporativa y sin instalaciones adicionales. El OCR productivo queda desacoplado: por defecto usa `tesseract` si existe como CLI; si no existe, devuelve texto vacio. El backend Tesseract actual corre varias pasadas con resize, contraste, binarizacion y distintos PSM. Para evaluacion offline se puede usar el `ocr_text` del CSV, pero solo en `train`, `validation` y `test_internal`. El golden debe evaluarse productivamente desde imagen cuando haya OCR disponible.

La funcion publica `diagnose_image(picture_url)` usa el preset candidato cuando las dependencias opcionales estan instaladas: EasyOCR + reglas/fuzzy + Naive Bayes textual + heuristica visual + clasificador MobileNetV3. Si EasyOCR o Torch/Torchvision no estan instalados, el servicio degrada al backend disponible sin romper el contrato.

Tambien existe un backend opcional fuerte con EasyOCR. No es dependencia base por peso de instalacion, pero se puede probar con:

```bash
python -m venv .venv-ocr
.venv-ocr/bin/python -m pip install ".[strong-ocr]"
PYTHONPATH=src .venv-ocr/bin/python scripts/evaluate_end_to_end.py --ocr-backend easyocr --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --output-json outputs/reports/end_to_end_validation_easyocr.json
```

## Comandos

Preparar entorno desde cero:

```bash
make setup-candidate
```

Los caches locales no se incluyen en el proyecto. `outputs/image_cache/` se crea automaticamente al evaluar imagenes desde URL.

## Datos y artefactos locales

El repositorio no versiona datasets, modelos entrenados ni documentos generados pesados. Para trabajar localmente, conserva o ubica estos archivos en las rutas esperadas:

- `anexo_1_dataset.csv`
- `outputs/datasets/*.csv`
- `outputs/golden_set/*.csv`
- `outputs/models/*.{pt,npz}`
- `outputs/documents/`

Los manifiestos, metricas `.json`, scripts y codigo fuente si quedan versionados para poder regenerar los artefactos.

Validar codigo rapido:

```bash
make test
```

Smoke end-to-end rapido:

```bash
PYTHONPATH=src .venv-ocr/bin/python scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 2 --output-json outputs/reports/candidate_smoke_latest.json
```

Validar candidato medido:

```bash
make validate-candidate
```

Construir datasets:

```bash
python scripts/build_datasets.py
```

Entrenar modelo textual:

```bash
python scripts/train_model.py
```

Evaluar offline con OCR del dataset:

```bash
python scripts/evaluate_model.py --dataset-csv outputs/datasets/validation.csv --output-json outputs/reports/validation.json
```

Comparar baselines:

```bash
python scripts/compare_baselines.py outputs/datasets/validation.csv outputs/reports/validation_baseline_comparison.json
```

Evaluar end-to-end desde `picture_url`:

```bash
python scripts/evaluate_end_to_end.py --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --output-json outputs/reports/end_to_end_validation.json
```

Evaluar con OCR fuerte opcional:

```bash
PYTHONPATH=src .venv-ocr/bin/python scripts/evaluate_end_to_end.py --ocr-backend easyocr --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --output-json outputs/reports/end_to_end_validation_easyocr.json
```

Evaluar preset candidato completo:

```bash
PYTHONPATH=src .venv-ocr/bin/python scripts/evaluate_end_to_end.py --preset candidate --dataset-csv outputs/datasets/validation.csv --rows-per-class 50 --dedupe-group-key leakage_group_id --output-json outputs/reports/candidate_validation_100_unique_latest.json
```

Entrenar modelo visual liviano:

```bash
/Users/O001224/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/train_visual_model.py --rows-per-class 500 --validation-rows-per-class 200 --output-json outputs/reports/visual_model_validation.json
```

Entrenar fusion visual con MobileNetV3 preentrenado:

```bash
PYTHONPATH=src .venv-ocr/bin/python scripts/train_pretrained_visual_model.py --rows-per-class 500 --validation-rows-per-class 200 --output-json outputs/reports/pretrained_visual_validation.json
```

Entrenar fusion visual con CLIP/OpenCLIP:

```bash
PYTHONPATH=src .venv-ocr/bin/python scripts/train_clip_visual_model.py --rows-per-class 500 --validation-rows-per-class 200 --fusion-details-csv outputs/reports/end_to_end_validation_100_unique_easyocr.details.csv --output-json outputs/reports/clip_visual_validation.json --insecure-hf-download --disable-hf-xet
```

Diagnosticar una URL:

```bash
python scripts/diagnose_url.py "https://..."
```

## Archivos principales

- `scripts/generate_golden_set.py`: genera el golden set v1 desde el CSV legacy.
- `scripts/build_datasets.py`: genera `train`, `validation` y `test_internal`, excluyendo golden.
- `scripts/train_model.py`: entrena el Naive Bayes textual.
- `scripts/evaluate_model.py`: evalua un dataset y guarda metricas.
- `scripts/evaluate_end_to_end.py`: evalua el camino productivo desde `picture_url`.
- `scripts/train_visual_model.py`: entrena un clasificador visual liviano con features de imagen.
- `scripts/train_pretrained_visual_model.py`: entrena un clasificador lineal sobre embeddings MobileNetV3.
- `scripts/train_clip_visual_model.py`: entrena un clasificador lineal sobre embeddings CLIP/OpenCLIP.
- `scripts/compare_baselines.py`: compara regex, modelo textual y ensemble.
- `scripts/diagnose_url.py`: ejecuta la funcion productiva sobre una URL.
- `src/image_moderation_poc/config.py`: rutas, thresholds y ratios.
- `src/image_moderation_poc/datasets.py`: parsing, curado, split por grupo y manifiesto.
- `src/image_moderation_poc/policy.py`: normalizacion, fuzzy matching y evidencia explicable.
- `src/image_moderation_poc/visual.py`: senales visuales heuristicas de overlays y badges.
- `src/image_moderation_poc/text_model.py`: modelo Naive Bayes sin `sklearn`.
- `src/image_moderation_poc/image_io.py`: descarga de imagenes con `verify_ssl=False`.
- `src/image_moderation_poc/ocr.py`: interfaz OCR reemplazable y backend Tesseract CLI.
- `src/image_moderation_poc/detector.py`: servicio principal y `diagnose_image`.
- `src/image_moderation_poc/evaluation.py`: precision, recall, F1 y metricas por segmento.
- `tests/`: pruebas unitarias de matching, modelo, detector y leakage.

## Criterio de aceptacion

Estado candidato medido en `outputs/reports/candidate_validation_100_unique_latest.json`, sobre 100 grupos unicos balanceados de validation:

- precision: 0.9545
- recall: 0.8400
- F1: 0.8936
- p99: 2622 ms
- fallos de descarga/proceso: 0

Este es el modelo/estado elegido para la entrega actual. No cumple todavia 95/95, pero queda medible, explicable y extensible para el siguiente incremento.

La version candidata debe reportar en el golden:

- precision >= 95%
- recall >= 95%
- recall por segmento positivo
- tasa de falsos positivos por segmento negativo
- p99 de latencia por imagen <= 5000 ms
- evidencia legible por prediccion positiva
