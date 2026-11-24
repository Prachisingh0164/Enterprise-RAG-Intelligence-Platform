# Enterprise Knowledge Intelligence & Agentic RAG Platform

Upload PDF / DOCX / TXT, ask questions in chat, get answers **with document name and page citations**. Role-based access control,
hybrid search, reranking, query rewriting, an agent that searches again when evidence is weak, and hallucination detection.

```
Streamlit UI --> FastAPI (JWT auth, RBAC) --> LangGraph agent --> Qdrant (dense) + BM25 (keyword)   --> PostgreSQL (docs, chunks, memory, logs)
                                                rewrite -> hybrid retrieve + rerank -> grade
                                                  insufficient? -> broaden query -> retrieve again (max 2 attempts)
                                                  -> generate (LLM or extractive) -> faithfulness check -> safe fallback
```

## Repository structure
```
backend/
  rag/            ingest.py (parse + chunk)  store.py (Qdrant + BM25)  retrieval.py (rewrite, hybrid RRF, rerank)
                  agent.py (LangGraph)  evaluate.py  auth.py  api.py  db.py  sample_data.py  seed.py
  sample_docs/    SYNTHETIC sample documents (PDF, DOCX, TXT)     eval/eval_set.jsonl  labelled questions
  tests/          pytest suite     Dockerfile  docker-compose.yml  monitoring/prometheus.yml  Makefile
frontend/         app.py (Streamlit: chat, search, documents, analytics)   dashboard-demo.html (static demo, no backend)
```

## Feature map
| Requirement | Where |
|---|---|
| PDF/DOCX/TXT ingestion, page-aware sentence-window chunking | `rag/ingest.py` |
| Embeddings + Qdrant | `rag/store.py` (offline hashing embedder by default; `EMBED_BACKEND=st` for sentence-transformers) |
| Hybrid search (BM25 + dense, Reciprocal Rank Fusion) | `rag/retrieval.py::hybrid` |
| Reranking | `retrieval.py::rerank` (lexical by default; `RERANK_BACKEND=cross` for a cross-encoder) |
| Query rewriting + conversational memory | `retrieval.py::rewrite`, `messages` table, `/chat` |
| Agentic retrieval | `agent.py`: grade -> broaden -> retry (LangGraph conditional edges) |
| Citations (document + page) | answer markers `[n]` mapped to `citations` |
| Hallucination detection | `agent.py::check_faithfulness`: per-sentence support + number check; unfaithful LLM answers fall back to extractive |
| Access control | JWT login; roles `employee / hr / finance / admin`; filter applied inside Qdrant **and** BM25 |
| Evaluation | `rag/evaluate.py`: hit@k, MRR, page hit, answer correctness, faithfulness, RBAC leakage, out-of-scope abstention; optional RAGAS |
| Observability | `query_logs` table, Prometheus `/metrics`, Analytics tab |

## Run locally
```bash
cd backend
pip install -r requirements-dev.txt
python -m rag.seed                 # demo users + SYNTHETIC sample docs
uvicorn rag.api:app --reload       # http://localhost:8000/docs

cd ../frontend                     # new terminal
pip install -r requirements.txt
streamlit run app.py               # http://localhost:8501
```
Demo logins: `employee/emp123` · `hr_user/hr123` · `finance_user/fin123` · `admin/admin123` (change before any real use).
Try: as **employee** ask "What was Q3 revenue?" (blocked), then sign in as **finance_user** and ask again (answered, cited).

Optional LLM: set `GEMINI_API_KEY` or `OPENAI_API_KEY` (see `backend/.env.example`). Without a key, answers are extractive: the most relevant sentences quoted from your documents with citations.

## Evaluate
`cd backend && python -m rag.evaluate` (or the Analytics tab). Sample-data results are near-perfect because the synthetic corpus is small and easy;
the real test is your own 100-200 question set in `eval/eval_set.jsonl` over real documents (SEC EDGAR filings, arXiv papers).
Format: `{"question", "doc", "page", "keyword", "user_role", "restricted"}`.

## Docker and AWS
`cd backend && docker compose up --build` -> dashboard :8501, API :8000, Qdrant :6333, Prometheus :9090, Grafana :3000.
AWS outline: images to ECR -> ECS Fargate (api, dashboard) behind an ALB -> RDS PostgreSQL -> Qdrant Cloud or Qdrant on ECS/EC2 with EFS -> secrets in Secrets Manager.

## Tests
`cd backend && pytest -q`

## Not included / limits
Scanned PDFs (no OCR), true semantic embeddings by default (install `requirements-extra.txt` and set `EMBED_BACKEND=st`), RAGAS run (code present, not exercised here),
React frontend, SSO, per-document audit trail, and rate limiting. Demo credentials and the JWT default secret are for local use only.
