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
ORDER_29N_PDF_PATH = ORDER_29N_RAW_DIR / "29N.pdf"
ORDER_29N_TEXT_PATH = ORDER_29N_PROCESSED_DIR / "order_29n_text.jsonl"
ORDER_29N_CHUNKS_PATH = ORDER_29N_PROCESSED_DIR / "order_29n_chunks.jsonl"
MINING_STATS_PATH = PROCESSED_DATA_DIR / "mining_stats.json"

TARGET_COLUMN = "has_contraindications"
ID_COLUMN = "exam_row_id"


load_dotenv(PROJECT_ROOT / ".env")
