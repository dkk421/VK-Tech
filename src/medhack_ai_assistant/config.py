from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
PDF_29N_PATH = DATA_DIR / "29N.pdf"
FACTORS_PATH = KNOWLEDGE_DIR / "factors.json"
RULES_PATH = KNOWLEDGE_DIR / "contraindication_rules.json"
RULES_RAW_PATH = KNOWLEDGE_DIR / "rules_raw.json"
SYNONYMS_PATH = KNOWLEDGE_DIR / "synonyms_ru.json"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
EXPERT_SUBMISSION_PATH = PROJECT_ROOT / "submission_expert.csv"
HYBRID_SUBMISSION_PATH = PROJECT_ROOT / "submission_hybrid.csv"

TARGET_COLUMN = "has_contraindications"
ID_COLUMN = "exam_row_id"
TEXT_COLUMNS = (
    "assigned_harmful_factors",
    "specialist_conclusions",
)


@dataclass(frozen=True)
class ModelConfig:
    max_features: int = 30000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 3
    random_state: int = 42
    validation_size: float = 0.2
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_count: int = 91
    n_estimators: int = 150
    max_depth: int = 5
    learning_rate: float = 0.08
