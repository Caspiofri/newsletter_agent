"""Central paths and environment loading. Every module imports locations from here, never from the cwd."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
EVALS_DIR = ROOT / "evals"

CREDENTIALS_PATH = CONFIG_DIR / "credentials.json"
TOKEN_PATH = CONFIG_DIR / "token.json"
CHROMA_PATH = DATA_DIR / "chroma_db"
GOLDEN_SET_PATH = EVALS_DIR / "golden_set.json"
