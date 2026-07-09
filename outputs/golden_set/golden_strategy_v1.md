# Golden set v1 - moderacion de imagenes

## Proposito

Este golden set es un holdout de aceptacion para la POC de moderacion de imagenes. Debe usarse solo para evaluacion final y pruebas de regresion, nunca para entrenamiento, ajuste de prompts, tuning de umbrales, construccion de regex OCR o seleccion de modelos.

## Interpretacion de la fuente

El archivo fuente contiene salidas del moderador legacy: `picture_url`, `infraction_detected`, `labels_detected` y `ocr_text`. Como no se permite reetiquetado manual, este artefacto usa pseudo-etiquetas conservadoras de alta confianza:

- Positivo: `infraction_detected = true` en el legado con una etiqueta de politica no vacia e interpretable.
- Negativo: `infraction_detected = false` en el legado y OCR sin palabras similares a politica.
- Excluido: grupos duplicados o conflictivos, positivos sin etiqueta y negativos cuyo OCR contiene terminos similares a politica.

## Segmentacion

Las filas positivas se segmentan por familia de infraccion: campanas/eventos, promesas de entrega, precio/promocion, badges o prueba social de marketplace, claims de confianza/pago/plataforma, claims de calidad/originalidad y otras etiquetas legacy conocidas.

Las filas negativas se segmentan por complejidad de OCR/texto: sin texto OCR, texto corto no-politica, texto medio no-politica y texto denso no-politica.

## Controles anti-leakage

- `item_id`, extraido desde la URL de la imagen, es el grupo de leakage.
- El golden set contiene como maximo una fila por `item_id`.
- Se excluyen grupos con etiquetas fuente conflictivas.
- Las filas del golden estan marcadas con `do_not_train = true`.
- Durante evaluacion, la solucion debe recibir solo `picture_url`. Columnas como `ocr_text`, `labels_detected`, `infraction_detected`, `target_has_infraction` y `target_label` son etiquetas/metadatos de auditoria y no deben usarse como inputs del modelo.

## Composicion

Filas totales: 5000

Conteos por clase/segmento:

```
target_has_infraction  segment                       
False                  negative_dense_nonpolicy_text     500
                       negative_medium_nonpolicy_text    600
                       negative_no_ocr_text              700
                       negative_short_nonpolicy_text     700
True                   campaign_event                    250
                       marketplace_badge_social_proof    800
                       other_legacy_label                 50
                       price_promotion                   800
                       quality_originality_claim         100
                       shipping_promise                  350
                       trust_payment_platform_claim      150
```

Conteos por clase/sitio:

```
target_has_infraction  site
False                  CBT     1637
                       MCO       97
                       MLA      217
                       MLB      218
                       MLC       29
                       MLM      289
                       MLU       13
True                   CBT      227
                       MCO      186
                       MLA      551
                       MLB      975
                       MLC      274
                       MLM      242
                       MLU       45
```

## Uso como criterio de aceptacion

Ejecutar la funcion candidata sobre cada `picture_url` y comparar `has_infraction` contra `target_has_infraction`. Reportar precision y recall globales, ademas de recall por segmento positivo y tasa de falsos positivos por segmento negativo. Un modelo deberia aprobar solo si precision global >= 95%, recall global >= 95% y ningun segmento importante presenta una regresion severa escondida por las metricas agregadas.
