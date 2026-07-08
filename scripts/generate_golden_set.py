from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "anexo_1_dataset.csv"
OUTDIR = ROOT / "outputs" / "golden_set"
GOLDEN = OUTDIR / "golden_set_v1.csv"
AMBIGUOUS = OUTDIR / "excluded_ambiguous_policy_keyword_v1.csv"
PROFILE = OUTDIR / "golden_set_v1_profile.csv"
MANIFEST = OUTDIR / "golden_manifest_v1.json"
REPORT = OUTDIR / "golden_strategy_v1.md"

SEED = "meli-image-moderation-golden-v1"

POSITIVE_FAMILIES = [
    (
        "campaign_event",
        r"hot[\s_]?sale|black\s*friday|black\s*days|relampago|rel[aá]mpago",
    ),
    (
        "shipping_promise",
        r"envio\s+inmediato|env[ií]o\s+inmediato|entrega\s+inmediata|"
        r"pronta\s+entrega|mismo\s+d[ií]a|llega\s+hoy|llega\s+ma[ñn]ana|"
        r"chega\s+amanha|chega_amanha",
    ),
    (
        "price_promotion",
        r"\bgratis\b|envio\s+gratis|env[ií]o\s+gratis|\boferta\b|descuento|"
        r"promoci[oó]n|12\s+cuotas|cuotas\s+fijas",
    ),
    (
        "marketplace_badge_social_proof",
        r"recomendado|best[\s_]?seller|mercado\s+lider|mercado\s+l[ií]der|"
        r"official[\s_]?store|mas\s+vendido|m[aá]s\s+vendido|destacado|"
        r"#1\s+en\s+ventas|\bmeli\+|ultimas[\s_]+unidades|[uú]ltimas\s+unidades",
    ),
    (
        "trust_payment_platform_claim",
        r"compra\s+segura|mercado\s+pago|mercado\s+envios|mercado\s+env[ií]os|"
        r"m[eé]todos\s+de\s+pago|factura",
    ),
    (
        "quality_originality_claim",
        r"mejor\s+calidad|mejor\s+validaci[oó]n|nuevas\s+y\s+originales|"
        r"diferentes\s+colores",
    ),
]

POSITIVE_TARGETS = {
    "marketplace_badge_social_proof": 800,
    "price_promotion": 800,
    "shipping_promise": 350,
    "campaign_event": 250,
    "trust_payment_platform_claim": 150,
    "quality_originality_claim": 100,
    "other_legacy_label": 50,
}

NEGATIVE_TARGETS = {
    "negative_no_ocr_text": 700,
    "negative_short_nonpolicy_text": 700,
    "negative_medium_nonpolicy_text": 600,
    "negative_dense_nonpolicy_text": 500,
}


def stable_hash(value: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_url_columns(df: pd.DataFrame) -> pd.DataFrame:
    filename = df["picture_url"].str.rsplit("/", n=1).str[-1]
    extracted = filename.str.extract(
        r"D_NQ_NP_(?P<picture_number>\d+)-(?P<item_id>[A-Z]{3}\d+|CBT\d+)_(?P<image_period>\d{6})-F"
    )
    df = pd.concat([df, extracted], axis=1)
    df["site"] = df["item_id"].str[:3]
    df["leakage_group_id"] = df["item_id"].fillna(df["picture_url"])
    return df


def positive_family(labels: object) -> str:
    text = str(labels or "").strip().lower()
    if not text:
        return "legacy_positive_unlabeled"
    for family, pattern in POSITIVE_FAMILIES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return family
    return "other_legacy_label"


POLICY_KEYWORD_RE = re.compile(
    "|".join(pattern for _, pattern in POSITIVE_FAMILIES), flags=re.IGNORECASE
)


def negative_segment(ocr_text: object) -> str:
    text = "" if pd.isna(ocr_text) else str(ocr_text)
    if POLICY_KEYWORD_RE.search(text):
        return "ambiguous_policy_keyword_negative"
    length = len(text.strip())
    if length == 0:
        return "negative_no_ocr_text"
    if length <= 40:
        return "negative_short_nonpolicy_text"
    if length <= 180:
        return "negative_medium_nonpolicy_text"
    return "negative_dense_nonpolicy_text"


def allocate_by_site(pool: pd.DataFrame, total: int) -> pd.Series:
    counts = pool["site"].fillna("UNK").value_counts()
    raw = counts / counts.sum() * total
    alloc = raw.round().astype(int)
    for site, count in counts.items():
        if count > 0 and raw.loc[site] >= 0.5 and alloc.loc[site] == 0:
            alloc.loc[site] = 1
    diff = total - int(alloc.sum())
    if diff != 0:
        order = (raw - np.floor(raw)).sort_values(ascending=(diff < 0)).index.tolist()
        i = 0
        while diff != 0 and order:
            site = order[i % len(order)]
            if diff > 0:
                alloc.loc[site] += 1
                diff -= 1
            elif alloc.loc[site] > 0:
                alloc.loc[site] -= 1
                diff += 1
            i += 1
    return alloc.clip(upper=counts)


def stratified_sample(pool: pd.DataFrame, targets: dict[str, int]) -> pd.DataFrame:
    selected = []
    for segment, target in targets.items():
        segment_pool = pool[pool["segment"] == segment].copy()
        if segment_pool.empty:
            continue
        allocations = allocate_by_site(segment_pool, min(target, len(segment_pool)))
        for site, n in allocations.items():
            if n <= 0:
                continue
            site_pool = segment_pool[segment_pool["site"].fillna("UNK") == site].copy()
            selected.append(site_pool.sort_values("sample_hash").head(int(n)))
    if not selected:
        return pd.DataFrame(columns=pool.columns)
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df = parse_url_columns(df)

    url_conflicts = (
        df.groupby("picture_url")["infraction_detected"].nunique().loc[lambda s: s > 1].index
    )
    group_conflicts = (
        df.groupby("leakage_group_id")["infraction_detected"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    df["has_duplicate_picture_url"] = df.duplicated("picture_url", keep=False)
    df["has_label_conflict"] = df["picture_url"].isin(url_conflicts) | df[
        "leakage_group_id"
    ].isin(group_conflicts)

    df["positive_family"] = df["labels_detected"].apply(positive_family)
    df["ocr_len"] = df["ocr_text"].fillna("").astype(str).str.strip().str.len()
    df["negative_segment"] = df["ocr_text"].apply(negative_segment)
    df["segment"] = df.apply(
        lambda row: row["positive_family"]
        if bool(row["infraction_detected"])
        else row["negative_segment"],
        axis=1,
    )
    df["target_has_infraction"] = df["infraction_detected"].astype(bool)
    df["target_label"] = df["segment"]
    df["sample_hash"] = df["leakage_group_id"].apply(lambda v: stable_hash(str(v)))

    dedup = (
        df[~df["has_label_conflict"]]
        .sort_values(["sample_hash", "picture_url"])
        .drop_duplicates("leakage_group_id", keep="first")
        .copy()
    )

    positive_pool = dedup[
        (dedup["target_has_infraction"])
        & (dedup["segment"].isin(POSITIVE_TARGETS))
        & (dedup["labels_detected"].fillna("").astype(str).str.strip() != "")
    ].copy()
    negative_pool = dedup[
        (~dedup["target_has_infraction"])
        & (dedup["segment"].isin(NEGATIVE_TARGETS))
    ].copy()
    ambiguous = dedup[dedup["segment"] == "ambiguous_policy_keyword_negative"].copy()

    golden = pd.concat(
        [
            stratified_sample(positive_pool, POSITIVE_TARGETS),
            stratified_sample(negative_pool, NEGATIVE_TARGETS),
        ],
        ignore_index=True,
    ).sort_values(["target_has_infraction", "segment", "sample_hash"], ascending=[False, True, True])

    golden.insert(0, "golden_id", [f"GS-V1-{i:05d}" for i in range(1, len(golden) + 1)])
    golden["split"] = "golden_acceptance_v1"
    golden["annotation_method"] = "legacy_high_confidence_pseudo_label"
    golden["label_confidence"] = "high"
    golden["do_not_train"] = True

    output_columns = [
        "golden_id",
        "split",
        "do_not_train",
        "picture_url",
        "target_has_infraction",
        "target_label",
        "segment",
        "site",
        "item_id",
        "leakage_group_id",
        "image_period",
        "picture_number",
        "annotation_method",
        "label_confidence",
        "infraction_detected",
        "labels_detected",
        "ocr_text",
        "ocr_len",
        "has_duplicate_picture_url",
        "has_label_conflict",
    ]
    golden[output_columns].to_csv(GOLDEN, index=False)
    ambiguous[
        [
            "picture_url",
            "target_has_infraction",
            "segment",
            "site",
            "item_id",
            "image_period",
            "labels_detected",
            "ocr_text",
            "ocr_len",
        ]
    ].to_csv(AMBIGUOUS, index=False)

    profile = (
        golden.groupby(["target_has_infraction", "segment", "site"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["target_has_infraction", "segment", "rows"], ascending=[False, True, False])
    )
    profile.to_csv(PROFILE, index=False)

    manifest = {
        "version": "golden_set_v1",
        "source_csv": str(INPUT),
        "golden_csv": str(GOLDEN),
        "rows": int(len(golden)),
        "positive_rows": int(golden["target_has_infraction"].sum()),
        "negative_rows": int((~golden["target_has_infraction"]).sum()),
        "leakage_policy": {
            "group_key": "item_id parsed from picture_url, fallback picture_url",
            "one_row_per_group": True,
            "conflicting_picture_url_groups_excluded": int(len(url_conflicts)),
            "conflicting_item_groups_excluded": int(len(group_conflicts)),
            "golden_rows_marked_do_not_train": True,
            "model_input_for_evaluation": ["picture_url"],
            "label_generation_columns_not_allowed_as_model_features": [
                "infraction_detected",
                "labels_detected",
                "ocr_text",
                "target_label",
                "target_has_infraction",
            ],
        },
        "sampling": {
            "seed": SEED,
            "positive_targets": POSITIVE_TARGETS,
            "negative_targets": NEGATIVE_TARGETS,
            "secondary_stratification": "site",
            "excluded_from_acceptance": [
                "legacy_positive_unlabeled",
                "ambiguous_policy_keyword_negative",
                "label_conflicts",
                "duplicate groups beyond first selected item_id",
            ],
        },
        "source_profile": {
            "source_rows": int(len(df)),
            "source_positive_rows": int(df["infraction_detected"].sum()),
            "source_negative_rows": int((~df["infraction_detected"]).sum()),
            "unique_picture_urls": int(df["picture_url"].nunique()),
            "duplicate_picture_url_rows": int(df.duplicated("picture_url").sum()),
            "ambiguous_policy_keyword_negative_rows_after_group_dedup": int(len(ambiguous)),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    segment_counts = golden.groupby(["target_has_infraction", "segment"]).size().to_string()
    site_counts = golden.groupby(["target_has_infraction", "site"]).size().to_string()
    REPORT.write_text(
        f"""# Golden set v1 - moderacion de imagenes

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

Filas totales: {len(golden)}

Conteos por clase/segmento:

```
{segment_counts}
```

Conteos por clase/sitio:

```
{site_counts}
```

## Uso como criterio de aceptacion

Ejecutar la funcion candidata sobre cada `picture_url` y comparar `has_infraction` contra `target_has_infraction`. Reportar precision y recall globales, ademas de recall por segmento positivo y tasa de falsos positivos por segmento negativo. Un modelo deberia aprobar solo si precision global >= 95%, recall global >= 95% y ningun segmento importante presenta una regresion severa escondida por las metricas agregadas.
""",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
