import os, requests, pandas as pd, streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")
ss = st.session_state


def api(method, path, **kw):
    r = requests.request(method, API + path, headers={"Authorization": f"Bearer {ss.get('token', '')}"}, timeout=120, **kw)
    if r.status_code >= 400:
        st.error(r.json().get("detail", r.text)); st.stop()
    return r.json()


if "token" not in ss:
    st.title("Enterprise Knowledge Assistant")
    st.caption("Demo logins: employee/emp123 · hr_user/hr123 · finance_user/fin123 · admin/admin123")
    with st.form("login"):
        u, p = st.text_input("Username"), st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            r = requests.post(API + "/auth/login", json={"username": u, "password": p})
            if r.ok: ss.update(token=r.json()["access_token"], user=r.json()["username"], role=r.json()["role"], msgs=[], cid=None); st.rerun()
            else: st.error("Wrong username or password")
    st.stop()

with st.sidebar:
    st.write(f"**{ss.user}** · role: {ss.role}")
    if st.button("Sign out"): ss.clear(); st.rerun()
    if st.button("New conversation"): ss.msgs, ss.cid = [], None; st.rerun()

tabs = ["Chat", "Search", "Documents"] + (["Analytics"] if ss.role == "admin" else [])
T = dict(zip(tabs, st.tabs(tabs)))


def show_answer(a):
    st.markdown(a["answer"])
    if a.get("flagged_sentences"):
        st.warning("Possible unsupported statements: " + "; ".join(f["sentence"] for f in a["flagged_sentences"]))
    for c in a.get("citations", []):
        with st.expander(f"[{c['n']}] {c['doc_name']} · page {c['page']}"): st.write(c["snippet"])


with T["Chat"]:
    for m in ss.msgs:
        with st.chat_message(m["role"]):
            show_answer(m["data"]) if m["role"] == "assistant" else st.write(m["content"])
    if q := st.chat_input("Ask about your documents"):
        ss.msgs.append({"role": "user", "content": q})
        a = api("POST", "/chat", json={"question": q, "conversation_id": ss.cid}); ss.cid = a["conversation_id"]
        ss.msgs.append({"role": "assistant", "content": a["answer"], "data": a}); st.rerun()
    if ss.msgs and ss.msgs[-1]["role"] == "assistant":
        a = ss.msgs[-1]["data"]
        with st.expander("How the agent got this answer"):
            c = st.columns(4); c[0].metric("Faithfulness", f"{a['faithfulness']:.0%}"); c[1].metric("Latency", f"{a['latency_ms']:.0f} ms"); c[2].metric("Search attempts", a["attempts"]); c[3].metric("Tokens", a["tokens"])
            st.caption(f"Rewritten query: {a['rewritten_query']} · answer mode: {a['mode']}")
            st.dataframe(pd.DataFrame(a["steps"]), hide_index=True); st.dataframe(pd.DataFrame(a["retrieved"]), hide_index=True)

with T["Search"]:
    q = st.text_input("Search documents (hybrid keyword + semantic)")
    if q:
        for r in api("GET", "/search", params={"q": q, "k": 6}):
            with st.container(border=True):
                st.markdown(f"**{r['doc_name']}** · page {r['page']} · score {r['score']}  \nkeyword rank {r['sparse_rank']} · semantic rank {r['dense_rank']}"); st.write(r["text"])

with T["Documents"]:
    if ss.role != "employee":
        with st.form("up"):
            f = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
            roles = st.multiselect("Who can read it?", ["all", "employee", "hr", "finance", "admin"], default=["all"] if ss.role == "admin" else [ss.role])
            if st.form_submit_button("Upload") and f:
                d = api("POST", "/documents", files={"file": (f.name, f.getvalue())}, data={"allowed_roles": ",".join(roles) or "all"})
                st.success(f"Indexed {d['name']}: {d['pages']} pages, {d['n_chunks']} chunks")
    docs = api("GET", "/documents")
    st.dataframe(pd.DataFrame(docs)[["id", "name", "allowed_roles", "pages", "n_chunks", "uploaded_by", "created_at"]] if docs else pd.DataFrame(), hide_index=True, use_container_width=True)

if "Analytics" in T:
    with T["Analytics"]:
        s = api("GET", "/admin/stats"); c = st.columns(5)
        c[0].metric("Documents", s["documents"]); c[1].metric("Chunks", s["chunks"]); c[2].metric("Queries", s["queries"])
        if s["queries"]:
            c[3].metric("Avg latency", f"{s['avg_latency_ms']} ms"); c[4].metric("Avg faithfulness", f"{s['avg_faithfulness']:.0%}")
            c = st.columns(4); c[0].metric("p95 latency", f"{s['p95_latency_ms']} ms"); c[1].metric("Abstain rate", f"{s['abstain_rate']:.0%}"); c[2].metric("Agent retries", f"{s['retry_rate']:.0%}"); c[3].metric("Tokens used", s["total_tokens"])
            l, r = st.columns(2); l.subheader("Queries per day"); l.bar_chart(pd.Series(s["daily_queries"])); r.subheader("Most cited documents"); r.bar_chart(pd.Series(s["top_cited_documents"]))
            st.subheader("Recent queries"); st.dataframe(pd.DataFrame(s["recent"]), hide_index=True)
        st.subheader("Retrieval and answer quality")
        if st.button("Run evaluation"):
            m = api("POST", "/admin/eval"); st.json(m)
