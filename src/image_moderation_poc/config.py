from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT_DIR / "anexo_1_dataset.csv"
GOLDEN_CSV = ROOT_DIR / "outputs" / "golden_set" / "golden_set_v1.csv"
DATASET_DIR = ROOT_DIR / "outputs" / "datasets"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "text_nb_model_v1.npz"
VISUAL_MODEL_PATH = MODEL_DIR / "visual_feature_model_v1.npz"
PRETRAINED_VISUAL_MODEL_PATH = MODEL_DIR / "pretrained_visual_linear_v1.npz"
IMAGE_CACHE_DIR = ROOT_DIR / "outputs" / "image_cache"

RANDOM_SEED = "meli-image-moderation-poc-v1"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_INTERNAL_RATIO = 0.15

DEFAULT_RULE_THRESHOLD = 0.88
DEFAULT_MODEL_THRESHOLD = 0.58
DEFAULT_HYBRID_THRESHOLD = 0.62
CANDIDATE_VISUAL_FUSION_THRESHOLD = 0.91
