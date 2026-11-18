import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app import config, gemini
from app.retriever import Retriever

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever()
    config.LOG_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="KYC Policy RAG Service", lifespan=lifespan)

PROMPT = """You are a banking compliance assistant. Answer the question using ONLY the context below.
Cite sources inline as [1], [2] matching the context numbers.
Be concise: use short bullet points and keep the answer under 150 words.
If the context does not contain the answer, say "I could not find this in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


def cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok * config.PRICE_INPUT_PER_1M + out_tok * config.PRICE_OUTPUT_PER_1M) / 1_000_000


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    hits = state["retriever"].search(req.question, req.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    if not hits:
        raise HTTPException(404, "No documents indexed")

    context = "\n\n".join(
        f"[{i}] ({h['source']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits, start=1)
    )
    gen = gemini.generate(PROMPT.format(context=context, question=req.question))

    total_ms = (time.perf_counter() - t_start) * 1000
    cost = cost_usd(gen["input_tokens"], gen["output_tokens"])
    tps = gen["output_tokens"] / gen["gen_seconds"] if gen["gen_seconds"] > 0 else 0

    metrics = {
        "total_ms": round(total_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(gen["gen_seconds"] * 1000, 1),
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "tokens_per_sec": round(tps, 1),
        "cost_usd": round(cost, 6),
        "retries": gen["retries"],
    }
    sources = [
        {"n": i, "source": h["source"], "page": h["page"], "score": round(h["score"], 3), "snippet": h["text"][:200]}
        for i, h in enumerate(hits, start=1)
    ]

    with open(config.LOG_FILE, "a") as f:
        f.write(json.dumps({"ts": time.time(), "question": req.question, **metrics}) + "\n")

    return {"answer": gen["text"], "sources": sources, "metrics": metrics}


@app.get("/metrics/summary")
def summary():
    """p50/p95 latency and cost per 1k requests from the log (retried requests excluded)."""
    if not config.LOG_FILE.exists():
        return {"requests": 0}
    rows = [json.loads(l) for l in config.LOG_FILE.read_text().splitlines() if l.strip()]
    clean = [r for r in rows if r.get("retries", 0) == 0] or rows
    lat = np.array([r["total_ms"] for r in clean])
    cost = np.array([r["cost_usd"] for r in clean])
    return {
        "requests_logged": len(rows),
        "requests_without_retries": len(clean),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "avg_cost_usd": round(float(cost.mean()), 6),
        "cost_per_1k_requests_usd": round(float(cost.mean() * 1000), 3),
    }