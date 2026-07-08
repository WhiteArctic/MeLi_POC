from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from image_moderation_poc import config


POSITIVE_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("campaign_event", r"hot[\s_]?sale|black\s*friday|black\s*days|relampago|rel[aá]mpago"),
    (
        "shipping_promise",
        r"envio\s+inmediato|env[ií]o\s+inmediato|entrega\s+inmediata|pronta\s+entrega|"
        r"mismo\s+d[ií]a|llega\s+hoy|llega\s+ma[ñn]ana|chega\s+amanha|chega_amanha",
    ),
    (
        "price_promotion",
        r"\bgratis\b|envio\s+gratis|env[ií]o\s+gratis|\boferta\b|descuento|"
        r"promoci[oó]n|12\s+cuotas|cuotas\s+fijas",
    ),
    (
        "marketplace_badge_social_proof",
        r"recomendado|best[\s_]?seller|mercado\s+lider|mercado\s+l[ií]der|"
        r"official[\s_]?store|mas\s+vendido|m[aá]s\s+vendido|destacado|#1\s+en\s+ventas|"
        r"\bmeli\+|ultimas[\s_]+unidades|[uú]ltimas\s+unidades",
    ),
    (
        "trust_payment_platform_claim",
        r"compra\s+segura|mercado\s+pago|mercado\s+envios|mercado\s+env[ií]os|"
        r"m[eé]todos\s+de\s+pago|factura",
    ),
    (
        "quality_originality_claim",
        r"mejor\s+calidad|mejor\s+validaci[oó]n|nuevas\s+y\s+originales|diferentes\s+colores",
    ),
)

POLICY_KEYWORD_RE = re.compile(
    "|".join(pattern for _, pattern in POSITIVE_FAMILY_PATTERNS), flags=re.IGNORECASE
)


def stable_hash(value: str, seed: str = config.RANDOM_SEED) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def parse_url_columns(df: pd.DataFrame) -> pd.DataFrame:
    filename = df["picture_url"].str.rsplit("/", n=1).str[-1]
    extracted = filename.str.extract(
        r"D_NQ_NP_(?P<picture_number>\d+)-(?P<item_id>[A-Z]{3}\d+|CBT\d+)_(?P<image_period>\d{6})-F"
    )
    out = pd.concat([df.copy(), extracted], axis=1)
    out["site"] = out["item_id"].str[:3]
    out["leakage_group_id"] = out["item_id"].fillna(out["picture_url"])
    return out


def positive_family(labels: object) -> str:
    text = "" if pd.isna(labels) else str(labels).strip().lower()
    if not text:
        return "legacy_positive_unlabeled"
    for family, pattern in POSITIVE_FAMILY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return family
    return "other_legacy_label"


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


def prepare_source_frame(source_csv: Path = config.SOURCE_CSV) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
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
    df["has_label_conflict"] = df["picture_url"].isin(url_conflicts) | df[
        "leakage_group_id"
    ].isin(group_conflicts)
    df["positive_family"] = df["labels_detected"].apply(positive_family)
    df["negative_segment"] = df["ocr_text"].apply(negative_segment)
    df["segment"] = df.apply(
        lambda row: row["positive_family"]
        if bool(row["infraction_detected"])
        else row["negative_segment"],
        axis=1,
    )
    df["target_has_infraction"] = df["infraction_detected"].astype(bool)
    df["target_label"] = df["segment"]
    df["ocr_len"] = df["ocr_text"].fillna("").astype(str).str.strip().str.len()
    return df


def build_curated_training_pool(source: pd.DataFrame, golden: pd.DataFrame) -> pd.DataFrame:
    golden_groups = set(golden["leakage_group_id"].astype(str))
    pool = source[~source["leakage_group_id"].astype(str).isin(golden_groups)].copy()
    pool = pool[~pool["has_label_conflict"]].copy()

    positive = pool[
        (pool["target_has_infraction"])
        & (pool["labels_detected"].fillna("").astype(str).str.strip() != "")
        & (pool["segment"] != "legacy_positive_unlabeled")
    ]
    negative = pool[
        (~pool["target_has_infraction"])
        & (pool["segment"] != "ambiguous_policy_keyword_negative")
    ]
    curated = pd.concat([positive, negative], ignore_index=True)
    curated["stratum"] = (
        curated["target_has_infraction"].astype(str)
        + "|"
        + curated["segment"].astype(str)
        + "|"
        + curated["site"].fillna("UNK").astype(str)
    )
    return curated


def assign_group_splits(curated: pd.DataFrame) -> pd.DataFrame:
    groups = (
        curated.sort_values("picture_url")
        .groupby("leakage_group_id", as_index=False)
        .first()[["leakage_group_id", "stratum"]]
    )
    groups["split_hash"] = groups["leakage_group_id"].apply(lambda v: stable_hash(str(v)))
    assignments = []
    for _, stratum_groups in groups.groupby("stratum", dropna=False):
        ordered = stratum_groups.sort_values("split_hash").copy()
        n = len(ordered)
        train_end = int(round(n * config.TRAIN_RATIO))
        validation_end = train_end + int(round(n * config.VALIDATION_RATIO))
        split = ["train"] * train_end
        split += ["validation"] * max(validation_end - train_end, 0)
        split += ["test_internal"] * max(n - len(split), 0)
        ordered["split"] = split[:n]
        assignments.append(ordered[["leakage_group_id", "split"]])
    return pd.concat(assignments, ignore_index=True)


def build_dataset_splits(
    source_csv: Path = config.SOURCE_CSV,
    golden_csv: Path = config.GOLDEN_CSV,
    output_dir: Path = config.DATASET_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = prepare_source_frame(source_csv)
    golden = pd.read_csv(golden_csv)
    curated = build_curated_training_pool(source, golden)
    assignments = assign_group_splits(curated)
    curated = curated.merge(assignments, on="leakage_group_id", how="inner")

    columns = [
        "split",
        "picture_url",
        "target_has_infraction",
        "target_label",
        "segment",
        "site",
        "item_id",
        "leakage_group_id",
        "image_period",
        "picture_number",
        "infraction_detected",
        "labels_detected",
        "ocr_text",
        "ocr_len",
        "has_label_conflict",
    ]

    paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test_internal": output_dir / "test_internal.csv",
        "golden": golden_csv,
    }
    for split in ("train", "validation", "test_internal"):
        curated[curated["split"] == split][columns].to_csv(paths[split], index=False)

    manifest = {
        "source_csv": str(source_csv),
        "golden_csv": str(golden_csv),
        "split_group_key": "leakage_group_id",
        "ratios": {
            "train": config.TRAIN_RATIO,
            "validation": config.VALIDATION_RATIO,
            "test_internal": config.TEST_INTERNAL_RATIO,
        },
        "rows": {split: int((curated["split"] == split).sum()) for split in paths if split != "golden"},
        "groups": {
            split: int(curated.loc[curated["split"] == split, "leakage_group_id"].nunique())
            for split in paths
            if split != "golden"
        },
        "golden_groups_excluded_from_training": int(golden["leakage_group_id"].nunique()),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths

