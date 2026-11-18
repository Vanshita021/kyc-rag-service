import json
import faiss
import numpy as np
from app import config, gemini


class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(config.INDEX_DIR / "index.faiss"))
        self.chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text())

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        q = np.array(gemini.embed([query], "RETRIEVAL_QUERY"), dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append({**c, "score": float(score)})
        return results
