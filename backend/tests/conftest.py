import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/t.db"
os.environ["QDRANT_PATH"] = ":memory:"
for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "QDRANT_URL"): os.environ.pop(k, None)
from rag import seed
seed.run()
