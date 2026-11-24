"""Evaluation: retrieval (hit@k, MRR, page hit), answer correctness, faithfulness, RBAC leakage, abstention.
python -m rag.evaluate            (built-in metrics, offline)
python -m rag.evaluate --ragas    (adds RAGAS faithfulness / answer relevancy; needs `pip install ragas datasets` and an LLM key)"""
import json, sys
from . import config
from .agent import run_agent
from .auth import roles_for
from .db import Session, Document
from .sample_data import OUT_OF_SCOPE


def run(k=5):
    items = [json.loads(l) for l in config.EVAL_FILE.read_text().splitlines() if l.strip()]
    with Session() as s: docs = {d.name: d.allowed_roles for d in s.query(Document)}
    items = [i for i in items if i["doc"] in docs]
    if not items: raise RuntimeError("No evaluation documents ingested. Run: python -m rag.seed")
    rows = []
    for it in items:
        st = run_agent(it["question"], [], roles_for(it["user_role"]))
        names = [c["doc_name"] for c in st["contexts"]]
        rank = names.index(it["doc"]) + 1 if it["doc"] in names[:k] else 0
        rows.append(dict(**it, rank=rank, page_hit=any(c["doc_name"] == it["doc"] and c["page"] == it["page"] for c in st["contexts"][:k]),
                         correct=it["keyword"].lower() in st["answer"].lower(), faithfulness=st["faithfulness"], answer=st["answer"],
                         cited=[(c["doc_name"], c["page"]) for c in st["citations"]]))
    n = len(rows); restricted = [r for r in rows if r["restricted"]]
    leaks = 0   # employees must never receive restricted documents
    for r in restricted:
        st = run_agent(r["question"], [], roles_for("employee"))
        leaks += any(docs[c["doc_name"]] != ["all"] for c in st["contexts"] + st["citations"] if c["doc_name"] in docs)
    abst = sum(run_agent(q, [], roles_for("admin"))["abstained"] for q in OUT_OF_SCOPE)
    return {"questions": n, f"hit@{k}": round(sum(r["rank"] > 0 for r in rows) / n, 3), "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
            "page_hit_rate": round(sum(r["page_hit"] for r in rows) / n, 3), "answer_correctness": round(sum(r["correct"] for r in rows) / n, 3),
            "avg_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3), "rbac_leaks": leaks, "rbac_checks": len(restricted),
            "abstained_out_of_scope": f"{abst}/{len(OUT_OF_SCOPE)}", "failures": [r["question"] for r in rows if not r["correct"]][:10]}


def ragas_eval(limit=20):   # NOT exercised in this repo's tests: needs ragas + LLM key; API varies by ragas version
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    items = [json.loads(l) for l in config.EVAL_FILE.read_text().splitlines()][:limit]; q, a, c = [], [], []
    for it in items:
        st = run_agent(it["question"], [], roles_for(it["user_role"])); q.append(it["question"]); a.append(st["answer"]); c.append([x["text"] for x in st["contexts"]])
    return evaluate(Dataset.from_dict({"question": q, "answer": a, "contexts": c}), metrics=[faithfulness, answer_relevancy])


if __name__ == "__main__":
    from . import runtime; runtime.init()
    print(json.dumps(run(), indent=2))
    if "--ragas" in sys.argv: print(ragas_eval())
