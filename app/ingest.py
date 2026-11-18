"""Build the FAISS index from PDFs in data/raw.

Usage: python -m app.ingest
"""
import json
import faiss
import numpy as np
from pypdf import PdfReader
from app import config, gemini
import time


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def load_chunks() -> list[dict]:
    chunks = []
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.RAW_DIR}")
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append({"id": len(chunks), "source": pdf.name, "page": page_no, "text": piece})
        print(f"{pdf.name}: {len(reader.pages)} pages")
    return chunks


def main():
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    vectors = []
    for i in range(0, len(chunks), config.EMBED_BATCH):
        batch = [c["text"] for c in chunks[i:i + config.EMBED_BATCH]]
        vectors.extend(gemini.embed(batch, "RETRIEVAL_DOCUMENT"))
        print(f"Embedded {min(i + config.EMBED_BATCH, len(chunks))}/{len(chunks)}")
        time.sleep(config.EMBED_SLEEP)

    mat = np.array(vectors, dtype="float32")
    faiss.normalize_L2(mat)               # cosine similarity via inner product
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "index.faiss"))
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    print(f"Saved index ({mat.shape[0]} vectors, dim {mat.shape[1]})")


if __name__ == "__main__":
    main()
