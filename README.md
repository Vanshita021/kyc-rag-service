# KYC Policy RAG Service

A retrieval-augmented question-answering API over the **RBI Master Direction on Know Your Customer (KYC)**. Ask a compliance question and get a short answer with **inline citations and page numbers**, plus per-request latency, token and cost metrics.

Built with Python, FastAPI, FAISS and the Gemini API. I work on banking KYC systems (account opening, NSDL/CKYC/eKYC integrations), so I built this to explore how LLMs can help with regulatory lookups.

## How it works

```
PDF(s) --> extract text per page --> chunk (1000 chars, 150 overlap)
       --> embed (Gemini embeddings, 3072-dim) --> FAISS index

Question --> embed --> top-4 chunks (cosine similarity)
         --> prompt with numbered context --> Gemini --> answer with [1], [2] citations
                                                      --> metrics logged to JSONL
```

| Component | Choice |
|---|---|
| API | FastAPI (`POST /ask`, `GET /health`, `GET /metrics/summary`) |
| Vector search | FAISS flat inner-product index on normalised vectors |
| LLM / embeddings | Gemini API (models set via `.env`) |
| Corpus | RBI KYC Master Direction (107 pages, 321 chunks) |

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows (use source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
cp .env.example .env                                # then set GEMINI_API_KEY
# put the PDF(s) in data/raw/
python -m app.ingest                                # builds data/index
uvicorn app.main:app --reload                       # open http://127.0.0.1:8000/docs
python -m evals.retrieval_eval                      # run the retrieval eval
```

## Results

### Retrieval quality (page-verified)
30 hand-written questions, each with the page(s) of the PDF that answer it (checked against the document text). A hit means one of the top 4 retrieved chunks comes from an expected page.

| Metric | Result |
|---|---|
| Hit@4 | **28/30 (93%)** |
| Hit@1 | **23/30 (77%)** |
| Direct questions | 26/28 |
| Paraphrased questions | 2/2 |

Questions and expected pages: `evals/questions.json`. Per-question output: `evals/results.json`.

### Latency, tokens and cost
Same question, before and after turning off model "thinking" and asking for concise answers:

| | Before | After | Change |
|---|---|---|---|
| Total latency | 30,262 ms | 6,619 ms | -78% |
| Generation time | 29,081 ms | 3,625 ms | -88% |
| Output tokens | 1,649 | 204 | -88% |
| Est. cost / request | $0.0044 | $0.0008 | -81% |

*Caveat: this is a single request per setting, not a median over many. Cost is estimated from the list prices in `.env`. Only retry-free requests are counted for latency (see below).*

## What broke (and what I changed)

1. **Hidden "thinking" tokens.** The first answer used 1,649 output tokens for a ~200-token answer and took 30 s. Disabling thinking and asking for concise answers cut tokens by 88%.
2. **Misleading latency.** One run showed 54 s because the retry wrapper slept 40 s after a rate-limit error. I now time only the successful attempt, log a `retries` count, and exclude retried requests from `/metrics/summary`.
3. **Free-tier limits.** Embedding hit a 100-requests/minute cap during ingestion (fixed with smaller batches and a pause between them) and generation hit a daily per-model cap (the retry logic now fails fast on daily limits instead of sleeping).
4. **PDF extraction quality.** Text comes out with broken words ("th e") and chunks are cut mid-word because I split by character count. The eval compares text with whitespace removed to cope with this.
5. **Noise pages.** The cover and contents pages show up in the top 4 for some questions. They should be excluded at ingestion.
6. **Exact-term miss.** "When must Form 60 be obtained?" retrieved a page that only mentions Form 60 in passing, not the page that answers it. Short exact terms are weak for embeddings.
7. **Ground truth ambiguity.** "What is V-CIP?" matches both the definition (p.13) and the requirements (p.28-31). Counted as a miss against the definition page; the question should be split.

## Limitations
- Only retrieval is evaluated automatically; answer faithfulness is spot-checked by hand.
- 30 questions from one document and one annotator, so the numbers are indicative, not a benchmark.
- Runs locally; not deployed. The request log is a local file.
- Not legal or compliance advice. Always check the source text.

## Next steps
- Skip cover/contents/appendix pages at ingestion and measure the change.
- Hybrid search (keyword + embeddings) and re-ranking to fix exact-term and Hit@1 misses.
- Sentence-aware chunking.
- Add an LLM-judged faithfulness eval.
- Agent layer: a supervisor that routes KYC checks to a document-check agent, a risk agent and this service as a policy-lookup tool (synthetic data only).