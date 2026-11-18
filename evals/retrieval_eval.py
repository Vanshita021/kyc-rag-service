"""Page-verified retrieval eval.

A question is a HIT if any of the top-K retrieved chunks comes from one of the
expected pages listed in evals/questions.json. Uses only the embedding API
(no generation quota).

Run from the project folder:  python -m evals.retrieval_eval
"""
import json
import time
from pathlib import Path
from app.retriever import Retriever

K = 4
QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))


def main():
    r = Retriever()
    rows = []
    scored = [q for q in QUESTIONS if q["expected_pages"]]  # skip out-of-scope questions
    hit_k = hit_1 = 0

    for item in scored:
        results = r.search(item["question"], K)
        pages = [h["page"] for h in results]
        rank = next((i for i, p in enumerate(pages, start=1) if p in item["expected_pages"]), None)
        hit_k += rank is not None
        hit_1 += rank == 1
        rows.append({"id": item["id"], "type": item["type"], "question": item["question"],
                     "expected": item["expected_pages"], "retrieved": pages, "rank": rank})
        print(f"{'HIT ' if rank else 'MISS'} rank={rank}  expected={item['expected_pages']}  got={pages}  [{item['id']}] {item['question'][:60]}")
        time.sleep(1)

    n = len(scored)
    print(f"\nPage-verified Hit@{K}: {hit_k}/{n} = {hit_k / n:.0%}   Hit@1: {hit_1}/{n} = {hit_1 / n:.0%}")
    for t in ("direct", "paraphrase"):
        sub = [x for x in rows if x["type"] == t]
        if sub:
            h = sum(x["rank"] is not None for x in sub)
            print(f"  {t}: {h}/{len(sub)}")

    Path("evals/results.json").write_text(
        json.dumps({"hit_at_k": hit_k / n, "hit_at_1": hit_1 / n, "k": K, "n": n, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()