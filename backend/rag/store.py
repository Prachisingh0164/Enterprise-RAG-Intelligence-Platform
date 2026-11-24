"""Vector store (Qdrant) + BM25 keyword index. Role filters are applied inside both retrievers."""
import math
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny, MatchValue, FilterSelector
from sklearn.feature_extraction.text import HashingVectorizer
from . import config
from .text import terms


class Embedder:
    """sentence-transformers when EMBED_BACKEND=st, otherwise an offline hashing embedder (lexical, no download)."""
    def __init__(self):
        if config.EMBED_BACKEND == "st":
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(config.EMBED_MODEL); self.dim = self.m.get_sentence_embedding_dimension()
        else:
            self.m, self.dim = None, 384
            self.hv = HashingVectorizer(n_features=384, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english")

    def encode(self, texts):
        if self.m: return self.m.encode(texts, normalize_embeddings=True).tolist()
        return self.hv.transform(texts).toarray().tolist()


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.N = k1, b, len(docs)
        self.dl = [len(d) for d in docs]; self.avg = (sum(self.dl) / self.N) if self.N else 1
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + .5) / (n + .5)) for t, n in df.items()}

    def scores(self, q):
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f: s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
            out.append(s)
        return out


class Store:
    def __init__(self):
        self.emb = Embedder(); self.chunks = {}; self.ids = []; self.bm25 = BM25([])
        if config.QDRANT_URL: self.q = QdrantClient(url=config.QDRANT_URL)
        elif config.QDRANT_PATH == ":memory:": self.q = QdrantClient(":memory:")
        else: self.q = QdrantClient(path=config.QDRANT_PATH)

    def _collection(self, recreate=False):
        if recreate and self.q.collection_exists(config.COLLECTION): self.q.delete_collection(config.COLLECTION)
        if not self.q.collection_exists(config.COLLECTION):
            self.q.create_collection(config.COLLECTION, vectors_config=VectorParams(size=self.emb.dim, distance=Distance.COSINE))

    def _rebuild(self):
        self.ids = list(self.chunks); self.bm25 = BM25([terms(self.chunks[i]["text"]) for i in self.ids])

    def _upsert(self, rows):
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]; vecs = self.emb.encode([r["text"] for r in batch])
            self.q.upsert(config.COLLECTION, points=[PointStruct(id=r["id"], vector=v, payload={k: r[k] for k in ("doc_id", "doc_name", "page", "roles", "text")}) for r, v in zip(batch, vecs)])

    def sync(self, rows):
        """Rebuild indexes from the database (the source of truth)."""
        self.chunks = {r["id"]: r for r in rows}; self._rebuild(); self._collection()
        if self.q.count(config.COLLECTION).count != len(rows):
            self._collection(recreate=True); self._upsert(rows)

    def add(self, rows):
        self._collection(); self._upsert(rows)
        self.chunks.update({r["id"]: r for r in rows}); self._rebuild()

    def remove_doc(self, doc_id):
        self._collection()
        self.q.delete(config.COLLECTION, points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])))
        self.chunks = {i: c for i, c in self.chunks.items() if c["doc_id"] != doc_id}; self._rebuild()

    def dense(self, query, roles, k):
        flt = Filter(must=[FieldCondition(key="roles", match=MatchAny(any=roles))])
        res = self.q.query_points(config.COLLECTION, query=self.emb.encode([query])[0], query_filter=flt, limit=k, with_payload=False).points
        return [(self.chunks[p.id], p.score) for p in res if p.id in self.chunks]

    def sparse(self, query, roles, k):
        sc = self.bm25.scores(terms(query)); rs = set(roles)
        ranked = sorted(((s, i) for s, i in zip(sc, self.ids) if s > 0 and rs & set(self.chunks[i]["roles"])), reverse=True)[:k]
        return [(self.chunks[i], s) for s, i in ranked]
