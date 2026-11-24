import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'rag.db'}")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_data"))   # ":memory:" for tests
COLLECTION = "enterprise_docs"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "lexical")
SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
SAMPLE_DIR = ROOT / "sample_docs"
EVAL_FILE = ROOT / "eval" / "eval_set.jsonl"
