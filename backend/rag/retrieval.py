"""Query rewriting, hybrid retrieval (dense + BM25, fused with RRF) and reranking."""
import re
from . import config, llm
from .runtime import get_store
from .text import terms

PRON = re.compile(r"\b(it|they|them|their|this|that|those|these|its|he|she)\b", re.I)


def rewrite(q, history):
    """Make follow-ups / vague questions standalone. LLM when available, otherwise a rule-based condenser."""
    q = q.strip()
    if not history and len(terms(q)) > 2: return q, 0, "none"
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    if llm.available() and (history or len(terms(q)) <= 2):
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        text, tok = llm.complete(f"Rewrite the final question as a standalone search query. Output only the query.\n\nConversation:\n{convo}\n\nQuestion: {q}")
        if text: return text.strip(), tok, "llm"
    if last_user and (len(terms(q)) <= 3 or PRON.search(q)):
        extra = [t for t in dict.fromkeys(terms(last_user)) if t not in terms(q)]
        return f"{q} {' '.join(extra)}", 0, "rule"
    return q, 0, "none"


def hybrid(query, roles, k=15, rrf_k=60):
    store = get_store(); fused = {}
    for name, hits in (("dense_rank", store.dense(query, roles, k)), ("sparse_rank", store.sparse(query, roles, k))):
        for rank, (c, _) in enumerate(hits, 1):
            e = fused.setdefault(c["id"], {**c, "rrf": 0.0, "dense_rank": None, "sparse_rank": None})
            e[name] = rank; e["rrf"] += 1 / (rrf_k + rank)
    return sorted(fused.values(), key=lambda x: -x["rrf"])


_cross = None


def rerank(query, cands, top_n=5):
    q = terms(query); qs, qb = set(q), set(zip(q, q[1:])); mx = max((c["rrf"] for c in cands), default=1) or 1
    for c in cands:
        ct = terms(c["text"]); cs, cb = set(ct), set(zip(ct, ct[1:]))
        c["coverage"] = len(qs & cs) / len(qs) if qs else 0.0
        bg = len(qb & cb) / len(qb) if qb else 0.0
        c["score"] = round(0.55 * c["coverage"] + 0.2 * bg + 0.25 * c["rrf"] / mx, 4)
    if config.RERANK_BACKEND == "cross" and cands:
        global _cross
        from sentence_transformers import CrossEncoder
        _cross = _cross or CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        for c, s in zip(cands, _cross.predict([(query, c["text"]) for c in cands])): c["score"] = round(float(s), 4)
    return sorted(cands, key=lambda c: -c["score"])[:top_n]
