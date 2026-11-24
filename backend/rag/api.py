import time, uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, make_asgi_app
import pandas as pd
from . import runtime, evaluate
from .agent import run_agent
from .auth import current_user, make_token, verify_pw, roles_for, Principal, ALL_ROLES
from .db import Session, User, Document, Chunk, Message, QueryLog
from .ingest import ingest
from .retrieval import hybrid, rerank

QUERIES = Counter("rag_queries_total", "Chat queries", ["abstained"])
LATENCY = Histogram("rag_latency_seconds", "End-to-end chat latency")


@asynccontextmanager
async def lifespan(app):
    runtime.init(); yield


app = FastAPI(title="Enterprise Knowledge Intelligence & Agentic RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/metrics", make_asgi_app())


class Login(BaseModel): username: str; password: str
class ChatIn(BaseModel): question: str; conversation_id: str | None = None


def admin_only(u: Principal = Depends(current_user)):
    if u.role != "admin": raise HTTPException(403, "Admin only")
    return u


@app.get("/health")
def health(): return {"ok": True}


@app.post("/auth/login")
def login(b: Login):
    with Session() as s:
        u = s.query(User).filter_by(username=b.username).first()
    if not u or not verify_pw(b.password, u.password_hash): raise HTTPException(401, "Wrong username or password")
    return {"access_token": make_token(u.username, u.role), "role": u.role, "username": u.username}


@app.get("/me")
def me(u: Principal = Depends(current_user)): return {"username": u.username, "role": u.role, "can_read_roles": roles_for(u.role)}


@app.post("/documents")
async def upload(file: UploadFile = File(...), allowed_roles: str = Form("all"), u: Principal = Depends(current_user)):
    if u.role == "employee": raise HTTPException(403, "Employees cannot upload documents")
    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()] or ["all"]
    if any(r not in ALL_ROLES for r in roles): raise HTTPException(400, f"roles must be from {ALL_ROLES}")
    if u.role != "admin" and not set(roles) <= {"all", u.role}: raise HTTPException(403, "You can only share with 'all' or your own role")
    try: return ingest(file.filename, await file.read(), roles, u.username)
    except ValueError as e: raise HTTPException(400, str(e))


@app.get("/documents")
def documents(u: Principal = Depends(current_user)):
    rs = set(roles_for(u.role))
    with Session() as s: return [d.to_dict() for d in s.query(Document).order_by(Document.id.desc()) if rs & set(d.allowed_roles)]


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: int, u: Principal = Depends(admin_only)):
    with Session() as s:
        d = s.get(Document, doc_id)
        if not d: raise HTTPException(404, "not found")
        s.query(Chunk).filter_by(doc_id=doc_id).delete(); s.delete(d); s.commit()
    runtime.get_store().remove_doc(doc_id); return {"deleted": doc_id}


@app.get("/search")
def search(q: str, k: int = 5, u: Principal = Depends(current_user)):
    top = rerank(q, hybrid(q, roles_for(u.role)), k)
    return [{k2: c[k2] for k2 in ("doc_name", "page", "text", "score", "coverage", "dense_rank", "sparse_rank", "rrf")} for c in top]


@app.post("/chat")
def chat(b: ChatIn, u: Principal = Depends(current_user)):
    cid = b.conversation_id or uuid.uuid4().hex[:12]
    with Session() as s:
        hist = [dict(role=m.role, content=m.content) for m in s.query(Message).filter_by(conversation_id=cid, username=u.username).order_by(Message.id).all()][-8:]
    t0 = time.time(); st = run_agent(b.question, hist, roles_for(u.role)); ms = round((time.time() - t0) * 1000, 1)
    LATENCY.observe(ms / 1000); QUERIES.labels(str(st["abstained"])).inc()
    with Session() as s:
        s.add_all([Message(conversation_id=cid, username=u.username, role="user", content=b.question), Message(conversation_id=cid, username=u.username, role="assistant", content=st["answer"])])
        s.add(QueryLog(username=u.username, user_role=u.role, question=b.question, rewritten=st["rewritten"], latency_ms=ms, tokens=st["tokens"], attempts=st["attempts"],
                       faithfulness=st["faithfulness"], n_flagged=len(st["flagged"]), abstained=st["abstained"], mode=st["mode"], cited_docs=sorted({c["doc_name"] for c in st["citations"]})))
        s.commit()
    return {"conversation_id": cid, "answer": st["answer"], "citations": st["citations"], "abstained": st["abstained"], "rewritten_query": st["rewritten"], "attempts": st["attempts"],
            "faithfulness": st["faithfulness"], "flagged_sentences": st["flagged"], "mode": st["mode"], "tokens": st["tokens"], "latency_ms": ms, "steps": st["steps"],
            "retrieved": [{k: c[k] for k in ("n", "doc_name", "page", "score", "dense_rank", "sparse_rank")} for c in st["contexts"]]}


@app.get("/admin/stats")
def stats(u: Principal = Depends(admin_only)):
    with Session() as s:
        logs = pd.DataFrame([{c.name: getattr(l, c.name) for c in QueryLog.__table__.columns} for l in s.query(QueryLog)])
        nd, nc = s.query(Document).count(), s.query(Chunk).count()
    out = {"documents": nd, "chunks": nc, "queries": len(logs)}
    if logs.empty: return out
    cited = pd.Series([d for ds in logs.cited_docs for d in (ds or [])]).value_counts().to_dict()
    return {**out, "avg_latency_ms": round(logs.latency_ms.mean(), 1), "p95_latency_ms": round(logs.latency_ms.quantile(.95), 1), "total_tokens": int(logs.tokens.sum()),
            "avg_faithfulness": round(logs.faithfulness.mean(), 3), "abstain_rate": round(float(logs.abstained.mean()), 3), "flagged_answers": int((logs.n_flagged > 0).sum()),
            "retry_rate": round(float((logs.attempts > 1).mean()), 3), "daily_queries": logs.created_at.dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict(),
            "queries_by_role": logs.user_role.value_counts().to_dict(), "top_cited_documents": cited,
            "recent": logs.sort_values("created_at", ascending=False).head(10)[["created_at", "username", "question", "latency_ms", "faithfulness", "abstained"]].astype(str).to_dict("records")}


@app.post("/admin/eval")
def run_eval(u: Principal = Depends(admin_only)):
    try: return evaluate.run()
    except RuntimeError as e: raise HTTPException(400, str(e))
