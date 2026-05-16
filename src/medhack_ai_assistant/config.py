from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
ORDER_29N_DIR = REFERENCE_DIR / "order_29n"
ORDER_29N_RAW_DIR = ORDER_29N_DIR / "raw"
ORDER_29N_PROCESSED_DIR = ORDER_29N_DIR / "processed"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"
ORDER_29N_PDF_PATH = ORDER_29N_RAW_DIR / "29N.pdf"
ORDER_29N_TEXT_PATH = ORDER_29N_PROCESSED_DIR / "order_29n_text.jsonl"
ORDER_29N_CHUNKS_PATH = ORDER_29N_PROCESSED_DIR / "order_29n_chunks.jsonl"
MINING_STATS_PATH = PROCESSED_DATA_DIR / "mining_stats.json"

TARGET_COLUMN = "has_contraindications"
ID_COLUMN = "exam_row_id"
TEXT_COLUMNS = (
    "assigned_harmful_factors",
    "specialist_conclusions",
)


load_dotenv(PROJECT_ROOT / ".env")


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


@dataclass(frozen=True)
class VLLMConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    chat_completions_path: str = os.getenv(
        "VLLM_CHAT_COMPLETIONS_PATH",
        "/chat/completions",
    )
    model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    api_key: str = os.getenv("VLLM_API_KEY", "EMPTY")
    timeout_seconds: float = float(os.getenv("VLLM_TIMEOUT_SECONDS", "90"))
    max_retries: int = int(os.getenv("VLLM_MAX_RETRIES", "2"))
    temperature: float = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("VLLM_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", "1800"))

    @property
    def chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        path = self.chat_completions_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base_url}{path}"
