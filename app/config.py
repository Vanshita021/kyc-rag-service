import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data" / "index"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "requests.jsonl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.8-flash")

PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.50"))

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 150
EMBED_BATCH = 20
EMBED_SLEEP = 15       # seconds between batches, keeps us under 100 texts/min
TOP_K = 4
