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

# Active collectors. Each can be overridden per profile in .env, e.g. AI_MEDIUM_TAGS=ai-agents,llm (comma-separated).
DEFAULT_MEDIUM_TAGS = ["ai-agents", "llm", "rag", "langchain", "generative-ai"]
DEFAULT_HN_QUERIES = ["LLM", "AI agents", "RAG", "LangGraph"]
HN_MIN_POINTS = 20
COLLECTOR_MAX_AGE_DAYS = 2

# Ranking pipeline: how many candidates per source reach the reader/ranker, and the minimum ranker score to be included.
SHORTLIST_QUOTA = {"inbox": 10, "hackernews": 6, "medium": 6}
READ_MAX_CHARS = 6000
RANK_MIN_SCORE = 5
DEDUPE_NEAR_MISS_MIN = 0.70  # cosine similarity; dedupe matches at or above this are reported in the trace
SNIPPET_SCORE_CAP = 6  # an article we could only read a teaser of cannot score above this

# LLM judge models (Claude, so the judge is not the same family as the ranker and writer)
JUDGE_MODEL_DAILY = "claude-haiku-4-5"  # per-run trace judge: cheap, runs every digest
JUDGE_MODEL_EVAL = "claude-sonnet-5-5"  # golden-set evals: stronger judge, run on demand
