from .db import Base, engine, Session, Chunk, Document, User
from .auth import hash_pw

_store = None
DEMO_USERS = [("admin", "admin123", "admin"), ("hr_user", "hr123", "hr"), ("finance_user", "fin123", "finance"), ("employee", "emp123", "employee")]


def get_store():
    global _store
    if _store is None:
        from .store import Store
        _store = Store()
    return _store


def init():
    Base.metadata.create_all(engine)
    with Session() as s:
        for u, p, r in DEMO_USERS:
            if not s.query(User).filter_by(username=u).first(): s.add(User(username=u, password_hash=hash_pw(p), role=r))
        s.commit()
        rows = [dict(id=c.id, doc_id=c.doc_id, doc_name=d.name, page=c.page, text=c.text, roles=d.allowed_roles)
                for c, d in s.query(Chunk, Document).filter(Chunk.doc_id == Document.id)]
    get_store().sync(rows)
