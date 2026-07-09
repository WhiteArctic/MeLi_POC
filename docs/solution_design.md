# Proyecto solucion - POC de moderacion de imagenes

## Objetivo

Construir una POC medible, explicable y extensible para detectar imagenes que infringen politicas comerciales por textos o badges superpuestos. La funcion productiva expuesta es:

```python
from image_moderation_poc import diagnose_image

diagnose_image(picture_url)
# {"has_infraction": bool, "evidence": str}
```

## Particion de datasets

Se usan cuatro conjuntos:

| Dataset | Uso | Puede entrenar? |
|---|---|---|
| `train.csv` | entrenamiento de modelo textual baseline | si |
| `validation.csv` | seleccion de umbrales y comparacion de variantes | no se usa para fit final |
| `test_internal.csv` | medicion interna previa al golden | no |
| `golden_set_v1.csv` | criterio final de aceptacion/regresion | nunca |

Reglas anti-leakage:

- La unidad de split es `leakage_group_id`, parseada desde el `item_id` de la URL.
- Los `leakage_group_id` del golden se excluyen antes de generar train/validation/test.
- Un grupo aparece en un solo split.
- `ocr_text`, `labels_detected`, `infraction_detected`, `target_label` y `target_has_infraction` no son inputs productivos. Son metadata de entrenamiento/evaluacion.

Particion generada:

| Split | Filas | Grupos |
|---|---:|---:|
| train | 377579 | 339189 |
| validation | 81255 | 72679 |
| test_internal | 81373 | 72690 |
| golden_set_v1 | 5000 | 5000 |

## Modelo entrenado en la POC

La POC entrena un modelo textual `TextNaiveBayes` sin dependencia de `sklearn`. Usa features hasheadas de palabras y char n-grams sobre OCR. Se eligio porque:

- corre en el entorno actual sin instalar librerias;
- es rapido, serializable y testeable;
- permite medir un baseline entrenable;
- deja estable el contrato para reemplazarlo por un modelo mas fuerte.

Este modelo usa `ocr_text` del CSV solo durante entrenamiento/evaluacion offline. En produccion, el texto debe venir de OCR ejecutado sobre la imagen descargada desde `picture_url`.

## Arquitectura productiva

```text
picture_url
  -> image_io.load_or_download_image(verify_ssl=False, cache local)
  -> OCRBackend.extract_text() multipass
  -> visual.VisualHeuristicDetector.detect()
  -> MobileNetV3VisualClassifier.predict_proba_image() opcional
  -> policy.find_policy_matches()
  -> TextNaiveBayes.predict_proba_one() full text + chunks
  -> ImageModerationService.diagnose_text()
  -> Diagnosis(has_infraction, evidence, scores)
```

Componentes estables:

- `ImageModerationService`: orquestacion y decision.
- `OCRBackend`: contrato para intercambiar `TesseractCliOCR`, PaddleOCR, EasyOCR o un servicio interno.
- `policy.py`: terminos, fuzzy matching y evidencia.
- `visual.py`: detector visual heuristico de overlays/badges como soporte.
- `evaluation.py`: metricas por clase y segmento.
- `datasets.py`: split y curado sin leakage.

Componentes reemplazables:

- `TextNaiveBayes`: baseline actual; reemplazable por modelo visual/textual mas fuerte.
- `TesseractCliOCR`: backend OCR local inicial.
- `EasyOCRBackend`: backend OCR fuerte opcional para mejorar cobertura sin API externa.
- `EnsembleOCR`: combinacion de OCRs para experimentacion de recall.
- `VisualHeuristicDetector`: heuristica liviana; reemplazable por detector visual entrenado.
- `MobileNetV3VisualClassifier`: clasificador visual opcional calibrado para fusion con EasyOCR.

## Descarga en red corporativa

`image_io.download_image` usa `urllib` con:

```python
ssl._create_unverified_context()
```

por defecto `verify_ssl=False`. Esto atiende el requerimiento de omitir certificados en red corporativa. En produccion real, se recomienda permitir configurar certificados corporativos en vez de desactivar verificacion globalmente.

## OCR, normalizacion y decision

`TesseractCliOCR` corre varias pasadas con imagen original, escala de grises 2x, contraste/sharpen 2x y binarizacion 2x. Usa un presupuesto total para mantener p99 bajo control.

`policy.py` normaliza acentos, mayusculas, guiones y ruido OCR. Tambien incluye sinonimos y correcciones para variantes frecuentes del OCR, como `official store`, `ofiatal sore`, `bestseller`, `amazon exclusive`, `desconto` y `super descontos`. Luego aplica:

- substring exacto para textos largos y casos claros;
- fuzzy matching con `difflib.SequenceMatcher` para textos cortos/medios;
- pesos por familia y decision con soporte para evitar que el modelo textual dispare positivos sin cues de politica.

Para controlar latencia, el fuzzy exhaustivo se limita a textos de hasta 80 tokens.

El `Decision Engine` combina:

- reglas textuales de alta precision;
- score semantico del Naive Bayes sobre texto completo y chunks/lineas OCR;
- senales visuales heuristicas de overlays/badges;
- abstencion cuando el modelo textual no tiene soporte de reglas, cues debiles o visual.

## Comparacion de baselines

Sobre validation:

| Baseline | Precision | Recall | F1 |
|---|---:|---:|---:|
| regex OCR | 1.000 | 0.876 | 0.934 |
| modelo textual, threshold 0.58 | 0.702 | 0.635 | 0.667 |
| regex OR modelo, threshold 0.99 | 0.823 | 0.974 | 0.892 |

Lectura: las reglas tienen precision muy alta pero no cubren todos los positivos; el modelo sube cobertura cuando se combina, pero introduce falsos positivos. Por eso la POC deja una arquitectura hibrida y medible, aunque el baseline liviano no debe venderse como solucion final 95/95.

Evaluacion end-to-end sobre 100 grupos unicos de validation, ejecutando `picture_url -> descarga -> OCR local -> politica -> decision`:

| Variante end-to-end | Precision | Recall | F1 | p99 |
|---|---:|---:|---:|---:|
| baseline inicial | 0.900 | 0.360 | 0.514 | 1677 ms |
| OCR multipass + politica | 0.955 | 0.420 | 0.583 | 4193 ms |
| normalizacion/sinonimos + policy engine | 0.966 | 0.560 | 0.709 | 4411 ms |
| pipeline conservador completo | 1.000 | 0.580 | 0.734 | 4207 ms |
| EasyOCR opcional | 1.000 | 0.680 | 0.810 | 2548 ms |
| Tesseract + EasyOCR ensemble | 1.000 | 0.740 | 0.851 | 6570 ms |
| MobileNetV3 embeddings + clasificador lineal | 0.919 | 0.285 | 0.435 | offline |
| EasyOCR + MobileNetV3 fusion | 0.955 | 0.840 | 0.894 | pendiente end-to-end integrado |
| Preset candidato integrado EasyOCR + MobileNetV3 | 0.955 | 0.840 | 0.894 | 2622 ms |
| CLIP/OpenCLIP embeddings | pendiente | pendiente | pendiente | bloqueado por descarga de pesos HF |

Lectura end-to-end: EasyOCR mejora recall y latencia frente a Tesseract multipass en la muestra medida, por lo que es el mejor backend OCR local probado. El ensemble recupera mas positivos, pero supera el limite de p99 <= 5000 ms. Como OCR fuerte tampoco llega al objetivo 95/95, el siguiente paso debe ser un modelo visual o multimodal entrenado.

Lectura visual/multimodal: las features visuales artesanales no alcanzan. MobileNetV3 preentrenado con clasificador lineal aporta senal complementaria y, fusionado con EasyOCR, sube recall a 0.84 manteniendo precision >= 0.95 en la muestra de 100 grupos. La corrida integrada `outputs/reports/candidate_validation_100_unique_latest.json` confirma precision 0.9545, recall 0.84, F1 0.8936, p99 2622 ms y 0 fallos. Aun asi no cumple 95/95; el siguiente incremento debe fine-tunear un backbone visual/multimodal o usar embeddings tipo CLIP/SigLIP con un clasificador entrenado y calibrado.

Se agrego `scripts/train_clip_visual_model.py` para probar CLIP/OpenCLIP con el mismo protocolo de muestreo y una fusion opcional con los resultados de EasyOCR. En este entorno, la descarga de `ViT-B-32/openai` fallo por certificados corporativos y luego por una respuesta incompleta de Hugging Face (26 KB contra 605 MB esperados), por lo que la metrica CLIP queda pendiente hasta contar con pesos cacheados o red habilitada.

## Camino para alcanzar 95/95

Para cumplir el reto tecnico, el siguiente incremento debe sumar un clasificador visual o multimodal:

1. Descargar/cachear imagenes de `train`, `validation` y `test_internal`.
2. Entrenar un clasificador visual liviano: EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny o SigLIP/CLIP pequeno.
3. Entrenar salida multi-label auxiliar por familia de infraccion.
4. Combinar:
   - score visual;
   - score OCR/modelo textual;
   - reglas fuzzy de alta precision.
5. Ajustar umbrales en validation.
6. Confirmar en test_internal.
7. Correr una sola vez contra golden como aceptacion.

## Lineamientos del reto cubiertos

- Funcion Python con `picture_url`: `diagnose_image`.
- Evidencia interpretable: terminos detectados, familia, score y estrategia.
- Costos: OCR local reemplazable, sin API externa en camino critico.
- Latencia: p99 end-to-end medido en muestra de validation; el pipeline conservador queda por debajo de 5000 ms en la muestra de 100 grupos unicos.
- Extensibilidad: nuevas politicas se agregan en `POLICY_TERMS` y/o familias de dataset.
- Monitoreo: usar precision/recall por segmento, drift de OCR, tasa de positivos, latencia p99 y top terminos detectados.
- Iteracion: falsos positivos/negativos de produccion entran a backlog, se etiquetan, se agregan a train/validation y se validan contra golden.
