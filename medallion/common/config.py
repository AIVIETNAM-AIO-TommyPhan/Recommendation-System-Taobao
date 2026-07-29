from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "input_data"
DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

RANDOM_SEED = 42
SMOOTHING = 20
TARGET = "click"

ID_COLS = ["userid", "adgroup_id", "campaign_id", "customer"]
SENTINEL_COLS = ["brand", "cms_segid", "pvalue_level", "new_user_class_level"]
