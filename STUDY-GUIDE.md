# RAGmind — Deep Study Guide

This document explains how the system works, file by file and concept by
concept, at the level you would need to defend it in a technical interview.
Read it next to the code. Every section maps to a real file. The goal is not
that you memorize it, but that you understand *why each decision was made*,
because the "why" is what interviewers probe.

A note on how to use this: read it once start to finish to get the shape,
then a second time with each source file open beside the matching section.
The margin between "I built a RAG demo" and "I can explain the engineering
tradeoffs in my retrieval system" is exactly this document.

---

## 0. The one-paragraph pitch (memorize the shape, not the words)

"RAGmind is an agentic retrieval-augmented generation system. It answers
questions from a document corpus using hybrid search, dense semantic
embeddings blended with BM25 keyword scoring, then a grader model judges
whether the retrieved context is sufficient. If it is not, the system rewrites
the query and retries, up to a bounded number of attempts. It generates
answers only from retrieved context, with inline citations, and refuses when
the context does not contain the answer. Critically, it ships with an
evaluation harness that measures retrieval recall, answer correctness,
hallucination rate, and citation validity over a golden test set, so I can
prove the system works and quantify improvements."

That paragraph hits every 2026 hiring keyword: agentic, RAG, hybrid search,
embeddings, BM25, grounding, citations, evaluation, hallucination. You should
be able to say it in your sleep and then go deep on any single clause.

---

## 1. The architecture, and why it is shaped this way

```
                    ┌─────────────────────────────┐
   user question →  │   app/streamlit_app.py      │   (thin UI, swappable)
                    └──────────────┬──────────────┘
                                   │ calls pipeline.answer()
                    ┌──────────────▼──────────────┐
                    │   core/pipeline.py          │   (the agent / controller)
                    │   the retrieve→grade→retry  │
                    │   loop lives here           │
                    └───┬─────────┬─────────┬─────┘
                        │         │         │
            ┌───────────▼──┐ ┌────▼─────┐ ┌─▼──────────┐
            │ retriever.py │ │grader.py │ │generator.py│
            │ hybrid search│ │ judge +  │ │ cited      │
            │ (dense+BM25) │ │ rewrite  │ │ answer     │
            └──────┬───────┘ └────┬─────┘ └─────┬──────┘
                   │              │             │
              documents.py    llm.py ◄──────────┘   (Gemini behind an
              (chunking)      (LLM access)            abstract interface)
```

The single most important architectural idea: **the `core/` package has no
idea Streamlit exists.** Search the core folder for the word "streamlit" and
you will find nothing. That is intentional. The intelligence is decoupled
from the interface. In an interview, when asked "how would you turn this into
a production API?", your answer is: "I would write a FastAPI route that
imports `RAGPipeline` and calls `.answer()`, and delete the Streamlit file.
Nothing in the engine changes, because the engine never depended on the UI."
That answer demonstrates you understand separation of concerns, which is a
senior-level instinct in a junior candidate.

Why this matters more than it looks: most tutorial projects put everything in
one `app.py` where the embedding code, the prompt, and the `st.button` calls
are tangled together. That code cannot be tested, reused, or re-deployed.
Yours can. That is the difference reviewers feel even when they cannot name
it.

---

## 2. config.py — why configuration is an object, not scattered constants

The whole system's tunable behavior lives in one `Config` dataclass. Three
reasons this is the right call:

1. **Testability.** Because config is an object you can construct, you can
   build two of them (`alpha=1.0` and `alpha=0.5`) and run the evaluation on
   each to compare. If these values were hard-coded globals scattered across
   files, that experiment would be impossible without editing source between
   runs. The eval harness depends on config being an object.

2. **Self-documentation.** Every knob has a comment explaining what it does
   and why the default was chosen. When an interviewer asks "why chunk size
   900?", the answer is in the file, not in your fading memory.

3. **Twelve-factor deployment.** `from_env()` lets environment variables
   override defaults, so the same code runs locally and in the cloud with no
   edits. That is standard production practice.

**Interview-bait knobs to understand cold:**
- `chunk_size` / `chunk_overlap`: see section 3.
- `hybrid_alpha`: see section 4. This is THE tuning experiment.
- `top_k_final`: how many chunks reach the LLM. More context is not always
  better, irrelevant chunks are noise that can degrade the answer and cost
  more tokens. 4 is a deliberate, modest default.
- `max_retrieval_attempts`: the safety cap on the agent loop (section 6).
- `temperature=0.2`: low on purpose. For factual grounded QA you want the
  model to stick to the context, not be "creative". High temperature is for
  brainstorming, not for citing security documents.

---

## 3. documents.py — chunking is the foundation everyone underestimates

RAG lives or dies on chunking, and almost nobody talks about it, which makes
it a great thing for you to talk about.

**Why chunk at all?** Two reasons. First, embedding models have a maximum
input length; you cannot embed a whole 50-page document into one vector
meaningfully, the meaning gets averaged into mush. Second, retrieval
precision: if the answer is one paragraph, you want to retrieve that
paragraph, not the entire document, so the LLM sees signal, not noise.

**Why overlap?** This is the subtle part. We cut documents into windows of
`chunk_size` characters. Imagine the sentence that answers a question sits
exactly at a chunk boundary. Without overlap, it gets sliced in half: chunk A
ends with "the mitigation is to use long, complex" and chunk B starts with
"service account passwords." Neither chunk contains the whole fact, so even if
retrieval picks the right chunk, the LLM sees a fragment. Overlap means the
last `overlap` (150) characters of each chunk are repeated at the start of the
next, so any sentence shorter than 150 chars always appears intact in at least
one chunk. The cost is mild redundancy; the benefit is you stop losing facts
at boundaries. This tradeoff, redundancy vs. boundary safety, is exactly the
kind of thing to mention unprompted.

**Why break on boundaries?** The code does not cut blindly at character 900;
it backs up to the nearest paragraph break, then sentence end, then space,
within a 250-char window. This keeps chunks human-readable, which matters
because we show them as citations. A chunk that starts mid-word looks broken
to a recruiter clicking "show sources."

**The Chunk dataclass and metadata.** Each chunk carries its `source`
(filename) and `chunk_idx`. This metadata is not decoration, it is what makes
citations possible later. The generator can say "[2] (source: mitre_attack.txt)"
only because every chunk remembers where it came from. Lose the metadata and
you lose traceability, which is a dealbreaker for any regulated industry.

---

## 4. retriever.py — hybrid search, the enterprise-grade core

This is the file to understand most deeply. It is also the most genuinely
impressive, because most candidates only know pure vector search.

### 4.1 The two retrievers and why you need both

**Dense (semantic) retrieval.** We use a sentence-transformer model to turn
each chunk into a 384-dimensional vector (an "embedding"). Texts with similar
*meaning* land near each other in this vector space, even if they share no
words. "How do I stop my password from being stolen" and "credential theft
mitigation" have almost no words in common but similar meaning, so their
vectors are close. At query time we embed the query and rank chunks by cosine
similarity.

- **Cosine similarity** is the cosine of the angle between two vectors. 1.0
  means they point the same direction (same meaning), 0 means perpendicular
  (unrelated). We `normalize_embeddings=True` so every vector has length 1,
  which means the dot product equals the cosine directly, and a dot product is
  just one fast matrix multiply (`self.embeddings @ q_vec`). That is a small
  but real performance decision.

**Sparse (keyword) retrieval, BM25.** Dense search has a famous weakness: it
blurs rare, exact tokens. Ask for "CVE-2021-44228" and a dense model may
return generically vulnerability-flavored chunks while missing the one chunk
that literally contains that ID, because the embedding does not preserve the
exact string. BM25 is the opposite: it scores by exact word overlap, weighted
so rare words matter more and long documents do not win just by length. So we
run both and fuse them. This is "hybrid search," and it is what Elastic,
Pinecone, and Weaviate ship by default in 2026 precisely because neither
retriever alone is enough.

### 4.2 BM25, explained term by term (we implemented it by hand)

We did not call a library for BM25, we wrote it, so you can explain every
piece. The score of a chunk for a query is the sum over query terms of:

```
idf(term) * ( f * (k1 + 1) ) / ( f + k1 * (1 - b + b * doclen/avgdl) )
```

- `f` is how many times the term appears in this chunk (term frequency). More
  occurrences → higher score.
- `idf(term)` is inverse document frequency: terms appearing in few chunks get
  a high weight, terms in every chunk (like "the") get near-zero weight. This
  is why BM25 is good at rare, meaningful tokens.
- `k1` (1.5) controls *saturation*: the 5th occurrence of a word helps less
  than the 1st. Without saturation, a chunk that spams a keyword would
  dominate unfairly.
- `b` (0.75) controls *length normalization*: `doclen/avgdl` is this chunk's
  length over the average. The `b` term penalizes long chunks so they do not
  win just by containing more words. b=0 would disable length normalization;
  b=1 applies it fully; 0.75 is the textbook compromise.

If asked "why not just use the library?", the honest answer: "For production
I would use a battle-tested implementation, but I implemented it myself here
so I actually understand the formula rather than treating it as a black box."
That is a maturity signal, not a naivety one.

### 4.3 Fusion, and the normalization trap

The two retrievers produce scores on totally different scales. Cosine is
roughly 0 to 1. BM25 is unbounded and can be 8 or more. You **cannot** just
add them, BM25 would swamp the cosine every time. So we min-max normalize each
score list to 0..1 first (`_minmax`), then blend:

```
fused = alpha * dense_norm + (1 - alpha) * sparse_norm
```

`alpha` (the `hybrid_alpha` config knob) is the dial between the two. This is
your headline evaluation experiment: run the eval at alpha=1.0 (pure
semantic), alpha=0.0 (pure keyword), and alpha=0.5 (hybrid), and show that
hybrid beats both on retrieval recall. That produces the single most valuable
sentence in your portfolio: "moving from pure vector search to hybrid raised
retrieval recall from X% to Y% on my test set." Evidence of an improvement you
made and measured.

One honest caveat to know: min-max normalization is sensitive to outliers
(one huge BM25 score compresses everything else toward 0). A more advanced
alternative is Reciprocal Rank Fusion (RRF), which fuses by rank position
instead of raw score and sidesteps the scale problem entirely. Knowing that
RRF exists and why someone might prefer it is a great thing to mention as
"what I would try next."

---

## 5. llm.py — the abstraction boundary, and a security point

### 5.1 Why an abstract base class for one provider

`LLMClient` is an abstract interface; `GeminiClient` implements it; `make_llm`
is a factory. We only use Gemini, so why the ceremony? Because the rest of the
engine, the grader, the generator, the pipeline, only ever references the
abstract `LLMClient`. None of them know or care that it is Gemini underneath.
To switch to Claude or a local model, you write one new class and change one
line in the factory. Drawing the abstraction boundary at "any chat LLM" rather
than hard-coding Gemini calls everywhere is the architectural-judgment move.
When an interviewer asks "how hard would it be to switch models?", you say
"one new class, one line", and that is a real answer backed by the code.

### 5.2 Why raw HTTP instead of the SDK

We call Gemini with Python's standard-library `urllib`, not the official SDK.
For learning, this is better: you can see the exact JSON the API expects
(`systemInstruction`, `contents`, `generationConfig`) and exactly how the
response is parsed. There is no magic. For a heavy production system you might
use the SDK for built-in retries and streaming, but for understanding and for
a dependency-light deploy, raw HTTP is clearer and you can defend every field.

### 5.3 The security point you must say out loud

`self.key = os.getenv("GEMINI_API_KEY")`. The key comes from the environment.
It is never written in the code and never committed to GitHub. This is the
same discipline as the serverless function on the portfolio website. For a
cybersecurity-focused candidate, this is not a throwaway, it is on-brand
evidence that you handle secrets correctly. Say it in interviews: "API keys
live in environment variables and secrets managers, never in source."

### 5.4 Defensive parsing

The client raises a clear error if Gemini returns no candidates or empty text,
rather than silently returning `""`. An empty string flowing downstream would
look like a hallucination or a broken answer with no explanation. Failing
loudly with a useful message is a production instinct, you want errors to be
diagnosable, not silent.

---

## 6. grader.py + pipeline.py — what makes this an AGENT

This is the pair that elevates the project from "RAG" to "agentic RAG", the
2026 frontier framing.

### 6.1 The grader: a model judging its own retrieval

After retrieving, before answering, we ask a cheap LLM a yes/no question: does
this context actually contain enough to answer the user? We force it to return
strict JSON (`{"sufficient": true/false, "reason": "..."}`) so we get a
machine-readable decision, not prose we have to regex. Demanding structured
output and parsing it robustly is itself a named, in-demand skill ("structured
outputs"). The `_extract_json` helper tolerates the model wrapping JSON in code
fences, because real models do that and a brittle parser breaks in production.

**Cost engineering:** the grader uses `gemini-2.5-flash`, the cheap fast model,
because grading is an easy classification task. We reserve model budget for the
final answer. Routing easy subtasks to cheap models and hard ones to expensive
models is a real cost lever production teams pull, and naming it shows you
think about more than just correctness.

**Fail-safe design:** if grading itself throws, `grade_context` defaults to
"sufficient = true" so the user still gets an answer rather than an error, and
it records the failure in the reason field so it is visible, not hidden.
Choosing which way to fail (open vs. closed) is a deliberate decision; here we
fail open because a slightly-less-vetted answer beats a hard error for the user.

### 6.2 The self-correction loop: perceive → judge → act → repeat

`pipeline.py` is the controller. The loop:

1. Retrieve for the current query.
2. Grade sufficiency.
3. If sufficient → generate cited answer, return.
4. If not → call `rewrite_query` to produce a better search query, loop.
5. Never exceed `max_retrieval_attempts`; if exhausted, answer with the best
   context available but set `low_confidence=True` so the UI warns the user.

**Why this is genuinely "agentic":** the system makes its own decision about
whether its work is good enough and takes corrective action without a human in
the loop. That perceive-judge-act-repeat cycle is the textbook definition of
an agent. It is not just an if-statement, because the "act" step (query
rewriting) uses an LLM to generate a genuinely different, hopefully better
search, often using terminology the user did not know ("my password got stolen"
→ "credential theft mitigation Kerberoasting").

**Why the attempt cap is the responsible part:** an unbounded agent loop is
how you get infinite loops and runaway API bills. Bounding it at 3, and
degrading gracefully to a low-confidence answer rather than failing, is
"bounded autonomy", the responsible-engineering version of agents. Say this
explicitly; safety-conscious framing of agentic systems is exactly what mature
teams want to hear in 2026.

### 6.3 The trace

Every loop iteration is recorded as a `Step` (attempt number, query used,
sources retrieved, grader verdict and reason). This `steps` list is not
decoration: it is *observability*. It is what lets the UI show "here is how I
reasoned," what lets you debug a bad answer ("oh, retrieval never found the
right doc on attempt 1"), and what the eval harness reads to compute
retrieval recall. Observability as a first-class concern, not an afterthought,
is a production hallmark.

---

## 7. generator.py — grounding, citations, and the anti-hallucination contract

The generator's system prompt enforces three rules that together are the
trustworthiness story:

1. **Answer only from the provided context.** Not from the model's training
   knowledge. This is what "grounded" means.
2. **Cite every claim** with a passage number like [1], [2].
3. **Refuse when the context lacks the answer**, with an exact sentence we can
   detect ("I don't have enough information in the provided sources...").

That third rule is the anti-hallucination contract and it is the most
important safety property. A model told "answer from this context and admit
when you cannot" fabricates far less than one handed the bare question. For
finance, healthcare, or government buyers, a system that confidently makes
things up is worse than useless, so measurable refusal behavior is a selling
point, not a limitation.

**Citation verification.** After the model answers, `_verify_citations` scans
for every `[k]` and keeps only those pointing at a real passage (1..N). Models
sometimes cite a `[5]` when only 4 passages were given, a subtle failure you
can only catch by checking. We then surface the genuinely-cited source files
to the UI so a human can click and verify. "Trust but verify" implemented in
code, not just promised in a prompt.

---

## 8. eval/evaluate.py — the part that gets you hired

Internalize this framing: **anyone can build a RAG demo; the rare skill is
proving it works.** This harness is the proof, and it is what most student
portfolios completely lack.

### 8.1 The golden set

`golden_set.json` is a fixed set of questions with known correct answers and
the document each should come from. Crucially it includes two categories:
- **answerable**: questions the corpus can answer (we check correctness +
  retrieval).
- **unanswerable**: questions the corpus deliberately cannot answer, like "what
  is the capital of France" (we check that the system *refuses*).

Including unanswerable cases is the sophisticated move. It directly measures
hallucination: a good system abstains on these; a bad one makes something up.

### 8.2 The four metrics, and what each isolates

1. **Retrieval recall:** did the expected source document show up in what we
   retrieved? This isolates the *retriever* from the *generator*. If recall is
   low, no LLM can save you, the right text never reached it. This is the
   metric your hybrid-alpha experiment moves.
2. **Answer correctness (LLM-as-judge):** an LLM compares the system answer to
   the reference answer and scores 0..1. Exact string matching is hopeless for
   prose, so "LLM-as-judge" is the 2026-standard way to grade free text at
   scale. You should know its weakness too: the judge is itself a model and can
   be wrong or biased, so for high-stakes evaluation you would spot-check
   judge scores against human labels.
3. **Hallucination rate:** on unanswerable questions, how often did the system
   answer instead of refusing? Lower is better. This is the headline safety
   number.
4. **Citation validity:** of answers that made claims, how many cited a real
   source? Trustworthiness, quantified.

### 8.3 The workflow that produces your portfolio sentence

Run `python -m eval.evaluate 1.0` (pure semantic), then
`python -m eval.evaluate 0.5` (hybrid). Compare the two `report.json` files.
The delta in retrieval recall is your evidence. The sentence "tuning hybrid
alpha from 1.0 to 0.5 improved retrieval recall from X% to Y% and cut
hallucination from A% to B% on an 11-question golden set" is worth more than
another five tutorial projects, because it shows you measure and improve, not
just build.

---

## 9. The honest limitations (knowing these is a strength)

Interviewers respect a candidate who can critique their own system. Be ready
with these:

- **Small corpus, small eval set.** 11 questions is a demonstration, not a
  rigorous benchmark. A real eval has hundreds of cases across difficulty
  tiers. Say so; do not oversell.
- **Brute-force retrieval.** We score every chunk for every query. Fine for
  hundreds of chunks; for millions you need an approximate nearest-neighbor
  index (FAISS, HNSW) or a vector database. You know the scaling path.
- **LLM-as-judge is imperfect.** It can disagree with humans. For production
  you would calibrate it against human labels.
- **Min-max fusion is outlier-sensitive.** RRF is the more robust alternative
  you would explore next.
- **No re-ranking stage.** Strong production systems add a cross-encoder
  re-ranker after retrieval for a precision boost. That is the obvious next
  upgrade.

Naming the next upgrades (RRF, ANN index, cross-encoder re-ranker, bigger
eval) shows you see past the current build, which is exactly the forward-looking
quality a progressive employer is screening for.

---

## 10. How to talk about it in 30 seconds, 2 minutes, and 10 minutes

- **30 seconds:** the one-paragraph pitch in section 0.
- **2 minutes:** pitch + "the part I am proudest of is the evaluation harness,
  because it let me prove that hybrid search beat pure vector search on my test
  set, and it measures hallucination directly by including questions the corpus
  cannot answer."
- **10 minutes:** walk the architecture diagram (section 1), then go deep on
  whichever they poke: hybrid fusion math (4.2-4.3), the agentic loop and why
  it is bounded (6.2), or the eval metrics and their weaknesses (8.2, 9).

You built it, you understand it, you can defend it, and you can say what you
would do next. That is the whole game.
