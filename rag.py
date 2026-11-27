"""Hybrid (dense + BM25) advice search over docs/ using local Qdrant."""
import os, json, uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    SparseVector, Prefetch, FusionQuery, Fusion, Modifier,
)
from fastembed import TextEmbedding, SparseTextEmbedding

DOCS_DIR = "docs"
DB_PATH = "qdrant_db"
COLLECTION = "advice"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
CHUNK_WORDS = 250
OVERLAP = 40
WEAK_SCORE = 0.01  # assumption: below this, advice is considered weak -> None

_dense = TextEmbedding(model_name=DENSE_MODEL)
_sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(path=DB_PATH)
    return _client


def _chunks(text):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = words[i:i + CHUNK_WORDS]
        if not chunk:
            break
        out.append(" ".join(chunk))
        i += CHUNK_WORDS - OVERLAP
        if i >= len(words):
            break
    return out


def _read_docs():
    docs = []
    for p in sorted(Path(DOCS_DIR).glob("*.txt")):
        lines = p.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        source = lines[0].strip()
        body = "\n".join(lines[1:])
        for ch in _chunks(body):
            docs.append((ch, source))
    return docs


def build_index(force=False):
    c = client()
    collections = [x.name for x in c.get_collections().collections]
    if COLLECTION in collections and not force:
        info = c.get_collection(COLLECTION)
        if info.points_count and info.points_count > 0:
            return info.points_count
    # recreate
    try:
        c.delete_collection(COLLECTION)
    except Exception:
        pass
    c.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    docs = _read_docs()
    points = []
    # embed in batches
    texts = [d[0] for d in docs]
    dense_vecs = list(_dense.embed(texts))
    sparse_vecs = list(_sparse.embed(texts))
    for (ch, src), dv, sv in zip(docs, dense_vecs, sparse_vecs):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dv.tolist() if hasattr(dv, "tolist") else list(dv),
                "sparse": SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
            },
            payload={"text": ch, "source": src},
        ))
    c.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def _dense_vec(q):
    return list(_dense.embed([q]))[0].tolist()


def _sparse_vec(q):
    sv = list(_sparse.embed([q]))[0]
    return SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def search(query, k=5, mode="hybrid"):
    c = client()
    # ensure index
    try:
        info = c.get_collection(COLLECTION)
        if not info.points_count:
            build_index()
    except Exception:
        build_index()

    if mode == "dense":
        res = c.query_points(collection_name=COLLECTION, query=_dense_vec(query),
                             using="dense", limit=k)
        hits = res.points
    elif mode == "bm25":
        res = c.query_points(collection_name=COLLECTION, query=_sparse_vec(query),
                             using="sparse", limit=k)
        hits = res.points
    else:
        res = c.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=_dense_vec(query), using="dense", limit=k * 2),
                Prefetch(query=_sparse_vec(query), using="sparse", limit=k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
        hits = res.points

    out = []
    for h in hits:
        payload = h.payload or {}
        out.append({"text": payload.get("text", ""), "source": payload.get("source", ""),
                    "score": float(h.score) if h.score is not None else 0.0})
    return out


def advice_for(disease_name):
    q = f"{disease_name} symptoms and management"
    hits = search(q, k=2, mode="hybrid")
    if not hits:
        return None
    if hits[0]["score"] < WEAK_SCORE:
        return None
    return hits[:2]


if __name__ == "__main__":
    n = build_index(force=True)
    print("Indexed chunks:", n)
    # quick test
    r = search("tomato early blight management", k=3, mode="hybrid")
    for h in r:
        print(round(h["score"], 4), h["source"], h["text"][:80].replace("\n", " "))
