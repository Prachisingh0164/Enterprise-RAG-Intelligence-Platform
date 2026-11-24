import io
import pytest
from fastapi.testclient import TestClient
from rag import evaluate
from rag.agent import run_agent, check_faithfulness
from rag.auth import roles_for
from rag.ingest import parse, chunk_pages
from rag.api import app

c = TestClient(app)


def token(u, p): return {"Authorization": "Bearer " + c.post("/auth/login", json={"username": u, "password": p}).json()["access_token"]}


def test_page_aware_chunking():
    ch = chunk_pages([(1, "A one. B two. C three."), (2, "D four.")], size=12)
    assert all(p in (1, 2) for p, _ in ch) and ch[-1][0] == 2 and len(ch) >= 3


def test_parsers_keep_pages():
    from rag import config
    for name, n in (("employee_handbook.pdf", 2), ("it_security_policy.docx", 2), ("product_manual.txt", 2)):
        assert len(parse(name, (config.SAMPLE_DIR / name).read_bytes())) == n


def test_answer_has_citation_with_page():
    st = run_agent("How many days of paid annual leave do full-time employees get?", [], roles_for("employee"))
    assert "24 days" in st["answer"] and st["citations"][0]["doc_name"] == "employee_handbook.pdf" and st["citations"][0]["page"] == 1 and st["faithfulness"] == 1.0


def test_rbac_blocks_restricted_documents():
    q = "What was Q3 revenue?"
    emp = run_agent(q, [], roles_for("employee")); fin = run_agent(q, [], roles_for("finance"))
    assert emp["abstained"] and not emp["citations"] and "48.2" in fin["answer"]


def test_follow_up_uses_memory_and_rewrites():
    h = [{"role": "user", "content": "How many days of paid annual leave do full-time employees get?"}, {"role": "assistant", "content": "24 days"}]
    st = run_agent("And how many can be carried over?", h, roles_for("employee"))
    assert "5 days" in st["answer"] and st["rewritten"] != "And how many can be carried over?"


def test_abstains_out_of_scope_and_agent_retries():
    st = run_agent("How do I bake sourdough bread?", [], roles_for("admin"))
    assert st["abstained"] and st["attempts"] == 2 and any("broaden" in s["step"].lower() for s in st["steps"])


def test_hallucination_detector_flags_unsupported():
    ctx = [{"text": "Full-time employees receive 24 days of paid annual leave."}]
    f, flagged = check_faithfulness("Employees receive 24 days of leave. They also get 99 days of sabbatical and free cars.", ctx)
    assert f < 1 and flagged and "99" in str(flagged)


def test_api_auth_upload_chat_stats():
    assert c.post("/chat", json={"question": "hi"}).status_code in (401, 403)
    emp, adm = token("employee", "emp123"), token("admin", "admin123")
    assert c.post("/documents", files={"file": ("x.txt", io.BytesIO(b"Hello world."))}, headers=emp).status_code == 403
    r = c.post("/documents", files={"file": ("lab_rules.txt", io.BytesIO(b"Lab coats must be worn at all times. Goggles are required near acid."))}, data={"allowed_roles": "all"}, headers=adm)
    assert r.status_code == 200 and r.json()["n_chunks"] == 1
    a = c.post("/chat", json={"question": "What is required near acid?"}, headers=emp).json()
    assert "goggles" in a["answer"].lower() and a["citations"][0]["doc_name"] == "lab_rules.txt"
    assert all(d["name"] not in ("finance_q3_report.pdf", "hr_compensation_confidential.pdf") for d in c.get("/documents", headers=emp).json())
    assert c.get("/admin/stats", headers=emp).status_code == 403 and c.get("/admin/stats", headers=adm).json()["queries"] >= 1


def test_evaluation_quality_gates():
    m = evaluate.run()
    assert m["hit@5"] >= 0.9 and m["answer_correctness"] >= 0.85 and m["rbac_leaks"] == 0 and m["avg_faithfulness"] >= 0.95 and m["abstained_out_of_scope"] == "5/5", m
