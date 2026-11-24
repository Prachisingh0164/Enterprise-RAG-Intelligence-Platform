"""Agentic RAG graph (LangGraph):
rewrite -> retrieve (hybrid + rerank) -> grade -> [insufficient: broaden -> retrieve again] -> generate -> verify -> [unfaithful: safe_answer]"""
import re, time
from typing import TypedDict
from . import llm
from .retrieval import rewrite, hybrid, rerank
from .text import terms, sentences, strip_markers

MAX_ATTEMPTS, MIN_COVERAGE = 2, 0.3
ABSTAIN = "I couldn’t find this in the documents you have access to."


class S(TypedDict, total=False):
    question: str; history: list; roles: list; rewritten: str; k: int; attempts: int
    contexts: list; sufficient: bool; answer: str; citations: list; faithfulness: float; flagged: list
    steps: list; tokens: int; mode: str; abstained: bool


def step(name):
    def deco(fn):
        def w(s):
            t0 = time.time(); upd, detail = fn(s)
            upd["steps"] = s.get("steps", []) + [{"step": name, "detail": detail, "ms": round((time.time() - t0) * 1000, 1)}]
            return upd
        w.__name__ = fn.__name__; return w
    return deco


@step("Query rewriting")
def rewrite_query(s):
    q, tok, how = rewrite(s["question"], s.get("history", []))
    return {"rewritten": q, "tokens": s.get("tokens", 0) + tok, "attempts": 0, "k": 15}, f"{how}: {q}"


@step("Hybrid retrieval + rerank")
def retrieve_docs(s):
    cands = hybrid(s["rewritten"], s["roles"], k=s["k"])
    top = rerank(s["rewritten"], cands, 5)
    for n, c in enumerate(top, 1): c["n"] = n
    return {"contexts": top, "attempts": s["attempts"] + 1}, f"{len(cands)} candidates -> top {len(top)}; best coverage {top[0]['coverage']:.0%}" if top else "no candidates"


@step("Relevance grading")
def grade_context(s):
    c = s["contexts"]; ok = bool(c) and c[0]["coverage"] >= MIN_COVERAGE and c[0]["coverage"] * max(len(set(terms(s["rewritten"]))), 1) >= 1
    return {"sufficient": ok}, "sufficient" if ok else f"insufficient (attempt {s['attempts']}/{MAX_ATTEMPTS})"


@step("Agent: broaden and search again")
def broaden_query(s):
    q = " ".join(dict.fromkeys(terms(s["question"])))
    return {"rewritten": q, "k": 30}, f"keyword-only query: {q}"


def extractive(question, contexts, max_sent=2):
    q = set(terms(question)); scored, seen = [], set()
    for c in contexts[:3]:
        for sent in sentences(c["text"]):
            ov = len(q & set(terms(sent)))
            if ov and sent not in seen:
                seen.add(sent); scored.append((ov / max(len(q), 1) + 0.02 * (5 - c["n"]), c["n"], sent))
    if not scored: return None
    scored.sort(key=lambda x: -x[0]); best = scored[0][0]
    picked = [x for x in scored if x[0] >= 0.6 * best][:max_sent]
    return " ".join(f"{t} [{n}]" for _, n, t in picked)


@step("Answer generation")
def generate_answer(s):
    if not s["sufficient"]:
        return {"answer": ABSTAIN, "abstained": True, "mode": "abstain", "citations": []}, "abstained: no sufficiently relevant context"
    ctx = "\n\n".join(f"[{c['n']}] ({c['doc_name']}, page {c['page']}) {c['text']}" for c in s["contexts"])
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in s.get("history", [])[-6:])
    text, tok = llm.complete("Answer ONLY from the context. Cite sources like [1]. If the context does not contain the answer, reply exactly: "
                             f"{ABSTAIN}\n\nConversation:\n{hist}\n\nContext:\n{ctx}\n\nQuestion: {s['question']}") if llm.available() else (None, 0)
    mode = "llm" if text else "extractive"
    text = text or extractive(s["question"], s["contexts"]) or ABSTAIN
    return {"answer": text.strip(), "mode": mode, "tokens": s.get("tokens", 0) + tok, "abstained": text.strip() == ABSTAIN}, f"mode={mode}"


def check_faithfulness(answer, contexts):
    """Hallucination detection: every answer sentence must be supported by retrieved text (term overlap + numbers must appear)."""
    ctx_terms = [set(terms(c["text"])) for c in contexts]; ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", " ".join(c["text"] for c in contexts)))
    flagged, total = [], 0
    for sent in sentences(strip_markers(answer)):
        st = set(terms(sent))
        if len(st) < 2: continue
        total += 1
        support = max((len(st & ct) / len(st) for ct in ctx_terms), default=0)
        bad = set(re.findall(r"\d+(?:\.\d+)?", sent)) - ctx_nums
        if support < 0.6 or bad: flagged.append({"sentence": sent, "support": round(support, 2), "unsupported_numbers": sorted(bad)})
    return (1.0 if not total else round((total - len(flagged)) / total, 3)), flagged


def finalize(s, ans, mode):
    f, flagged = check_faithfulness(ans, s["contexts"]) if not s.get("abstained") else (1.0, [])
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans)})
    cites = [c for c in s["contexts"] if c["n"] in used] or ([] if s.get("abstained") else s["contexts"][:1])
    return {"answer": ans, "mode": mode, "faithfulness": f, "flagged": flagged,
            "citations": [{"n": c["n"], "doc_name": c["doc_name"], "page": c["page"], "snippet": c["text"], "score": c["score"]} for c in cites]}, f


@step("Hallucination check")
def verify_answer(s):
    upd, f = finalize(s, s["answer"], s["mode"])
    return upd, f"faithfulness {f:.0%}; {len(upd['flagged'])} unsupported sentence(s)"


@step("Safe fallback: extractive answer")
def safe_answer(s):
    ans = extractive(s["question"], s["contexts"]) or ABSTAIN
    upd, f = finalize({**s, "abstained": ans == ABSTAIN}, ans, "extractive")
    return {**upd, "abstained": ans == ABSTAIN}, f"replaced unfaithful LLM answer; faithfulness {f:.0%}"


NODES = dict(rewrite_query=rewrite_query, retrieve_docs=retrieve_docs, grade_context=grade_context, broaden_query=broaden_query,
             generate_answer=generate_answer, verify_answer=verify_answer, safe_answer=safe_answer)
after_grade = lambda s: "generate_answer" if s["sufficient"] or s["attempts"] >= MAX_ATTEMPTS else "broaden_query"
after_verify = lambda s: "safe_answer" if s["faithfulness"] < 0.6 and s["mode"] == "llm" else "end"


def _build():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None
    g = StateGraph(S)
    for n, f in NODES.items(): g.add_node(n, f)
    g.set_entry_point("rewrite_query")
    g.add_edge("rewrite_query", "retrieve_docs"); g.add_edge("retrieve_docs", "grade_context")
    g.add_conditional_edges("grade_context", after_grade); g.add_edge("broaden_query", "retrieve_docs")
    g.add_edge("generate_answer", "verify_answer")
    g.add_conditional_edges("verify_answer", after_verify, {"safe_answer": "safe_answer", "end": END}); g.add_edge("safe_answer", END)
    return g.compile()


_graph = None


def run_agent(question, history, roles):
    global _graph
    if _graph is None: _graph = _build() or False
    st = {"question": question, "history": history, "roles": roles, "steps": [], "tokens": 0, "abstained": False}
    if _graph: return _graph.invoke(st)
    for n in ("rewrite_query", "retrieve_docs", "grade_context"): st.update(NODES[n](st))
    while after_grade(st) == "broaden_query":
        for n in ("broaden_query", "retrieve_docs", "grade_context"): st.update(NODES[n](st))
    for n in ("generate_answer", "verify_answer"): st.update(NODES[n](st))
    if after_verify(st) == "safe_answer": st.update(NODES["safe_answer"](st))
    return st
