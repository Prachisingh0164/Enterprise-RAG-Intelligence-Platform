"""Document ingestion: PDF / DOCX / TXT -> page-aware text -> sentence-window chunks -> DB + Qdrant."""
import io, re
from pathlib import Path
from .db import Session, Document, Chunk
from . import runtime


def parse(filename: str, data: bytes):
    """Return [(page_number, text)]. PDF = real pages, DOCX = explicit page breaks, TXT = form feeds."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(i + 1, re.sub(r"\s*\n\s*", " ", p.extract_text() or "")) for i, p in enumerate(PdfReader(io.BytesIO(data)).pages)]
    if ext == ".docx":
        import docx
        pages, cur = [], []
        for p in docx.Document(io.BytesIO(data)).paragraphs:
            if 'w:type="page"' in p._p.xml:
                pages.append(" ".join(cur)); cur = []
            if p.text.strip(): cur.append(p.text.strip())
        pages.append(" ".join(cur))
        return [(i + 1, t) for i, t in enumerate(pages)]
    if ext in (".txt", ".md"):
        return [(i + 1, re.sub(r"\s*\n\s*", " ", t)) for i, t in enumerate(data.decode("utf-8", "ignore").split("\f"))]
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_pages(pages, size=350, overlap=1):
    """Sentence-window chunking: never splits mid-sentence, keeps the page number, overlaps one sentence."""
    out = []
    for page, text in pages:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        cur = []
        for s in sents:
            if cur and sum(len(x) + 1 for x in cur) + len(s) > size:
                out.append((page, " ".join(cur))); cur = cur[-overlap:] if overlap else []
            cur.append(s)
        if cur: out.append((page, " ".join(cur)))
    return out


def ingest(filename: str, data: bytes, roles: list, uploader: str):
    pages = parse(filename, data)
    chunks = chunk_pages(pages)
    if not chunks: raise ValueError("No extractable text (scanned PDF? add OCR).")
    store = runtime.get_store()
    with Session() as s:
        for old in s.query(Document).filter_by(name=filename).all():       # re-upload replaces
            s.query(Chunk).filter_by(doc_id=old.id).delete(); store.remove_doc(old.id); s.delete(old)
        d = Document(name=filename, ext=Path(filename).suffix.lower(), allowed_roles=roles, pages=len(pages), n_chunks=len(chunks), uploaded_by=uploader)
        s.add(d); s.flush()
        objs = [Chunk(doc_id=d.id, page=p, idx=i, text=t) for i, (p, t) in enumerate(chunks)]
        s.add_all(objs); s.flush()
        rows = [dict(id=o.id, doc_id=d.id, doc_name=filename, page=o.page, text=o.text, roles=roles) for o in objs]
        s.commit(); out = d.to_dict()
    store.add(rows)
    return out
