import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class Retriever:
    def __init__(self, index_dir="index"):
        d = Path(index_dir)
        self.vectors = np.load(d / "vectors.npy")
        lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        self.chunks = [json.loads(l) for l in lines if l.strip()]
        self.model = SentenceTransformer(MODEL)

    @staticmethod
    def build(chunks, index_dir="index"):
        model = SentenceTransformer(MODEL)
        vecs = model.encode([c["text"] for c in chunks], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=True)
        d = Path(index_dir); d.mkdir(exist_ok=True)
        np.save(d / "vectors.npy", vecs.astype(np.float32))
        (d / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")

    def retrieve(self, query, k=6):
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]