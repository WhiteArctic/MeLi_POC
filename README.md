# Meli Image Moderation POC

POC reproducible para detectar infracciones en imagenes de publicaciones a partir de
`picture_url`. La salida publica del detector es:

```python
{
    "has_infraction": bool,
    "decision": str,
    "score": float,
    "evidence": str,
    "ocr_text": str,
    "signals": dict,
}
```

El foco del proyecto es ser entendible, medible y extensible: separa datos fuente,
splits, modelos, metricas y codigo productivo; evita leakage por item; y deja
evidencia legible para cada prediccion positiva.

## Que contiene el repositorio

- `src/image_moderation_poc/`: paquete Python con politicas, OCR, modelo textual,
  senales visuales, evaluacion y servicio principal.
- `scripts/`: comandos reproducibles para crear datasets, entrenar, evaluar y
  diagnosticar URLs individuales.
- `tests/`: pruebas unitarias de reglas, modelo, detector y splits sin leakage.
- `docs/solution_design.md`: diseno tecnico y decisiones de arquitectura.
- `docs/reproducibility.md`: guia para reconstruir artefactos desde un clone limpio.
- `docs/sharing.md`: checklist para compartir una copia limpia del proyecto.
- `outputs/**/*.json`: manifiestos y metricas livianas que documentan el estado medido.

El repositorio no versiona el CSV original, datasets derivados, pesos/modelos
binarios, caches de imagenes ni documentos generados.

## Prerequisitos

- Python 3.10 o superior.
- Acceso a internet para instalar dependencias y descargar imagenes desde
  `picture_url`.
- Para el candidato completo: dependencias opcionales de OCR/vision (`easyocr`,
  `torch`, `torchvision`) y descarga de pesos preentrenados.
- El archivo `anexo_1_dataset.csv` en la raiz del proyecto para reconstruir
  datasets/modelos.

El CSV esperado tiene estas columnas:

```text
picture_url,infraction_detected,labels_detected,ocr_text
```

## Quick start

Crear entorno base e instalar el paquete:

```bash
make setup
```

El entrypoint principal queda registrado como comando de consola:

```bash
image-moderation --help
```

Ejecutar pruebas:

```bash
make test
```

Reconstruir el golden set y los splits `train`, `validation` y `test_internal`:

```bash
make datasets
```

Entrenar el modelo textual base:

```bash
make train-text
```

Medir baselines offline usando el `ocr_text` del CSV:

```bash
make baselines
```

Ejecutar un smoke test end-to-end desde `picture_url`:

```bash
make smoke
```

## Candidato completo

El candidato medido usa EasyOCR, reglas/fuzzy matching, Naive Bayes textual,
heuristicas visuales y un clasificador visual MobileNetV3 opcional.

Instalar dependencias opcionales:

```bash
make setup-candidate
```

Entrenar el clasificador visual preentrenado:

```bash
make train-visual-pretrained
```

Evaluar el candidato sobre 100 imagenes balanceadas y deduplicadas por item:

```bash
make validate-candidate
```

Evaluar una muestra mayor de 500 imagenes balanceadas:

```bash
make validate-candidate-large
```

Diagnosticar una URL individual:

```bash
make diagnose URL="https://..."
```

La salida por defecto esta pensada para lectura humana y evaluacion manual:

```json
{
  "has_infraction": true,
  "decision": "infraction_detected",
  "score": 0.92,
  "evidence": "Se detectaron senales de politica...",
  "ocr_text": "Texto extraido de la imagen",
  "signals": {
    "rules": 0.75,
    "text_model": 0.88,
    "semantic_text": 0.91
  }
}
```

Para ver la metadata interna completa de depuracion:

```bash
make diagnose-debug URL="https://..."
```

Si se instalaron las dependencias opcionales del candidato (`make setup-candidate`),
se puede usar OCR/vision fuerte en una URL individual:

```bash
make diagnose-candidate URL="https://..."
```

## Flujo de datos

1. `golden_set_v1`: conjunto de examen. No se usa para entrenar, ajustar umbrales
   ni seleccionar modelos.
2. `train`: entrenamiento del modelo textual usando OCR legacy como etiqueta debil.
3. `validation`: seleccion de umbrales y comparacion de variantes.
4. `test_internal`: evaluacion interna antes de tocar el golden.

La particion se hace por `leakage_group_id`, derivado del `item_id` parseado desde
`picture_url`. Asi se evita que imagenes del mismo item caigan en conjuntos
distintos.

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

La funcion publica `diagnose_image(picture_url)` usa el preset candidato cuando
estan disponibles las dependencias opcionales. Si EasyOCR o Torch/Torchvision no
estan instalados, el servicio degrada al backend disponible sin romper el contrato.

## Artefactos y limpieza

Los artefactos pesados o sensibles se generan localmente y estan ignorados por Git:

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

Limpiar artefactos generados:

```bash
make clean-generated
```

Limpiar entornos y caches locales:

```bash
make clean-local
```

## Resultados de referencia

Estado candidato medido en `outputs/reports/candidate_validation_100_unique_latest.json`,
sobre 100 grupos unicos balanceados de `validation`:

- precision: 0.9545
- recall: 0.8400
- F1: 0.8936
- p99: 2622 ms
- fallos de descarga/proceso: 0

El estado actual es explicable y reproducible, pero no alcanza aun el objetivo
ideal de 95% precision y 95% recall. El siguiente incremento deberia enfocarse
en mejorar recall sin degradar precision, especialmente con OCR fuerte y mejores
senales visuales.
