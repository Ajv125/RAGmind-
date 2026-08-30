# ◆ RAGmind — Agentic Retrieval-Augmented Generation with Built-in Evaluation

An agentic RAG system that answers questions from a document corpus using
**hybrid search** (semantic + keyword), judges whether it retrieved enough to
answer, **self-corrects by rewriting the query and retrying** when it did not,
and answers **only from cited sources** — refusing when the answer is not in
the corpus. It ships with an **evaluation harness** that measures retrieval
recall, answer correctness, hallucination rate, and citation validity, so the
system's quality is proven, not asserted.

Primary corpus: cybersecurity reference material (NIST CSF 2.0, MITRE ATT&CK,
CVE advisories). The corpus is swappable — a second business/governance corpus
is included and selectable with zero code change.

---

## Why this is not just another chatbot

Most RAG demos do `retrieve → stuff into prompt → answer` and hope retrieval
was good. This system adds the things production teams actually need:

| Capability | What it does | Why it matters |
|---|---|---|
| **Hybrid search** | Blends dense embeddings with BM25 keyword scoring | Pure vector search misses exact tokens (CVE IDs, product names); hybrid is the enterprise default |
| **Agentic self-correction** | Grades retrieval; rewrites query and retries if insufficient | Recovers from bad initial retrieval instead of answering from thin context |
| **Grounded citations** | Answers cite the passages they came from; citations are verified | Trust and auditability — required in finance, healthcare, government |
| **Refusal on unknowns** | Says "I don't have enough information" instead of guessing | Directly reduces hallucination, the #1 enterprise AI risk |
| **Evaluation harness** | Measures recall, correctness, hallucination, citation validity | Proves the system works and quantifies improvements |

---

## Architecture

The engine (`core/`) is completely decoupled from the interface. There are
**zero UI imports in the core** — the Streamlit app is a thin shell that calls
`pipeline.answer()`. The same engine could be served by FastAPI or a CLI
without changing one line of core logic.

```
app/streamlit_app.py     UI shell (swappable)
core/pipeline.py         the agent: retrieve → grade → retry loop
core/retriever.py        hybrid search (dense embeddings + hand-written BM25)
core/grader.py           LLM judges sufficiency; rewrites failed queries
core/generator.py        grounded, cited answer generation
core/llm.py              provider-agnostic LLM client (Gemini behind an interface)
core/documents.py        document loading + overlapping chunking
core/config.py           every tunable knob, documented, in one place
eval/evaluate.py         the evaluation harness
eval/golden_set.json     test questions (answerable + deliberately unanswerable)
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key (free tier works)
export GEMINI_API_KEY="your-key-here"     # get one at aistudio.google.com

# 3. Run the app
streamlit run app/streamlit_app.py
```

First run downloads the embedding model (~80 MB) once, then caches it.

### Run the evaluation

```bash
python -m eval.evaluate 0.5      # run with hybrid alpha = 0.5
python -m eval.evaluate 1.0      # run with pure semantic search
# compare the two eval/report.json outputs to see hybrid's effect
```

---

## Example: the headline experiment

The evaluation harness exists to answer "is this actually good, and did my
changes help?" The intended experiment compares retrieval strategies:

- Pure semantic (`alpha=1.0`) vs. hybrid (`alpha=0.5`) on the same golden set
- Metric of interest: **retrieval recall** (did the right document get found?)
  and **hallucination rate** (did it refuse questions it should refuse?)

Run both and the report shows the tradeoff in hard numbers. (Run it yourself;
the numbers depend on your corpus and model version, and honest reporting of
your own measured results beats any number hard-coded in a README.)

---

## Tech stack

- **Python** — engine and orchestration
- **sentence-transformers** — dense embeddings (all-MiniLM-L6-v2, runs locally)
- **BM25 (implemented from scratch)** — sparse keyword retrieval
- **Google Gemini** — generation + grading (behind a swappable interface)
- **Streamlit** — UI shell
- **NumPy** — vector math

No vector database is required; retrieval is in-memory, which is the right
choice at this corpus size. The README's companion `STUDY-GUIDE.md` explains
the scaling path (FAISS/HNSW, re-ranking) when it would be needed.

---

## What I would build next

- **Reciprocal Rank Fusion** instead of min-max score fusion (more robust to
  outliers)
- **Cross-encoder re-ranker** after retrieval for a precision boost
- **Approximate nearest-neighbor index** (FAISS) to scale past in-memory search
- **Larger golden set** with difficulty tiers for a more rigorous benchmark

---

## Security note

API keys are read from environment variables only — never written in source,
never committed. AI features that accept user input are designed with prompt-
injection awareness. This reflects the same secure-handling discipline used
throughout my work.

See `STUDY-GUIDE.md` for a deep, file-by-file explanation of every design
decision and the concepts behind them.
