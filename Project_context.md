# MITAOE AI Assistant — Full Project Context

This document provides the COMPLETE architectural, technical, and engineering context for the MITAOE AI Assistant project.

The goal is to onboard a coding agent into the project with enough detail to continue development without making incorrect architectural assumptions.

This is NOT a toy chatbot project.

This is a production-oriented institutional RAG + retrieval system designed for:

* admissions assistance
* academic information retrieval
* faculty lookup
* placements information
* curriculum search
* event/club discovery
* future voice assistant integration

The architecture prioritizes:

* deterministic retrieval
* explainability
* metadata quality
* retrieval precision
* low hallucination rate
* inspectable pipelines

NOT:

* flashy AI features
* generic LangChain wrappers
* agent overengineering
* black-box retrieval

====================================================
PROJECT OBJECTIVE
=================

Build an institutional AI assistant for MITAOE capable of answering questions about:

* admissions
* eligibility
* placements
* faculty
* departments
* curriculum
* fees
* facilities
* clubs/events
* internships
* research
* notices

The assistant must:

* retrieve grounded answers
* cite institutional content
* avoid hallucinations
* work with noisy institutional data
* support future voice interfaces

====================================================
CURRENT PROJECT STATUS
======================

The project currently has:

✅ Website scraping completed
✅ CSV ingestion completed
✅ Document normalization pipeline
✅ Deduplication system
✅ Validation framework
✅ Semantic chunking pipeline
✅ BM25 retrieval layer
✅ Retrieval inspection UI
✅ Benchmark query framework
✅ Retrieval evaluation framework

NOT YET IMPLEMENTED:

* embeddings
* vector DB
* hybrid retrieval
* reranking
* conversational orchestration
* memory
* voice pipeline
* agentic workflows

====================================================
ARCHITECTURAL PHILOSOPHY
========================

The system is being built incrementally in layers.

Correct order:

```text id="jlwm45"
crawl
 ↓
clean
 ↓
normalize
 ↓
semantic structure
 ↓
chunking
 ↓
retrieval evaluation
 ↓
routing
 ↓
embeddings
 ↓
hybrid retrieval
 ↓
reranking
 ↓
LLM answering
 ↓
voice interface
```

We intentionally delayed embeddings because:

* bad chunks poison embeddings
* metadata quality matters first
* retrieval evaluation must exist first
* debugging vectors without inspection is painful

====================================================
DATA SOURCE
===========

Primary source:
MITAOE institutional website.

Original crawl exported into CSV.

CSV contains:

* URL
* HTML
* extracted text
* metadata

Initial problems included:

* duplicate pages
* boilerplate contamination
* inconsistent titles
* sidebar bleed
* footer contamination
* weak classifications
* short pages
* noisy template pages

====================================================
CURRENT DATA PIPELINE
=====================

Current flow:

```text id="jlwm46"
CSV
 ↓
ingestion pipeline
 ↓
cleaned documents
 ↓
semantic chunking
 ↓
BM25 retrieval
 ↓
retrieval inspection
```

====================================================
CURRENT DATASET OUTPUTS
=======================

Current important outputs:

```text id="jlwm47"
datasets/processed_documents.json
datasets/chunks.jsonl
reports/ingestion_report.json
reports/chunking_report.json
reports/bm25_benchmark_results.json
```

====================================================
INGESTION PIPELINE
==================

The ingestion pipeline already supports:

* deterministic title extraction
* structured metadata
* validation warnings
* quality warnings
* duplicate handling
* boilerplate stripping
* semantic sections
* summaries

====================================================
TITLE EXTRACTION
================

Priority order:

```text id="jlwm48"
og:title
 ↓
article h1
 ↓
main h1
 ↓
html title
 ↓
URL slug
```

Stored as:

```json id="jlwm49"
metadata.title_source
```

====================================================
VALIDATION SYSTEM
=================

Validation is warning-based, not rejection-heavy.

Hard failures ONLY for:

* invalid URL
* extraction exception
* empty content
* malformed row

Warnings include:

* low content
* weak classification
* title mismatch
* boilerplate heavy
* duplicate content

====================================================
DEDUPLICATION POLICY
====================

IMPORTANT:

Aggressive fuzzy deduplication was REMOVED.

We discovered:

* institutional pages share templates heavily
* fuzzy matching caused false-positive deletions

Current policy:
ONLY exact duplicates are removed.

No aggressive semantic deduplication.

False-positive deletion is considered more harmful than redundancy.

====================================================
CHUNKING SYSTEM
===============

Current chunking system exists and is modular.

Key features:

* semantic chunking
* FAQ-aware chunking
* faculty-profile chunking
* curriculum-aware chunking
* token-aware splitting
* overlap handling
* metadata propagation
* chunk validation
* chunk scoring hooks

====================================================
CURRENT CHUNK STATS
===================

Approximate current stats:

```text id="jlwm50"
documents_processed: 934
chunks_generated: 1893
avg_chunk_tokens: ~330
max_chunk_tokens: 1000
min_chunk_tokens: 40
```

Tiny chunks are already merged.

====================================================
CHUNKING RULES
==============

DO NOT:

* split blindly by character count
* use naive recursive splitters
* fragment FAQs
* split faculty records randomly
* split tables mid-way

Chunking priorities:

* semantic coherence
* independent understandability
* metadata preservation
* retrieval readiness

====================================================
CURRENT RETRIEVAL STACK
=======================

Current retrieval stack is BM25-only.

No embeddings yet.

Implemented:

* BM25 retrieval service
* retrieval inspection UI
* benchmark query runner
* chunk inspection
* metadata display
* query testing

====================================================
RETRIEVAL INSPECTOR
===================

Inspector currently supports:

* query input
* BM25 retrieval preview
* chunk inspection
* metadata inspection
* validation warnings
* source URL display

Purpose:
debugging retrieval quality.

====================================================
BENCHMARK FRAMEWORK
===================

Benchmark queries exist.

Current benchmark goal:
evaluate retrieval quality BEFORE embeddings.

Metrics planned:

* Recall@5
* Recall@10
* MRR
* Hit rate
* Precision@3

====================================================
KNOWN PROBLEMS
==============

Current major problems:

1. Unstable page classification
2. Weak semantic section typing
3. Admissions information buried inside blogs
4. No intent routing
5. Metadata inconsistency
6. Some content contamination still exists
7. Canonical pages not prioritized
8. BM25 searches entire corpus blindly

====================================================
CURRENT PHASE
=============

We are currently implementing:

# PHASE 2.5

Corpus normalization + retrieval routing layer

NOT embeddings yet.

====================================================
CURRENT PRIORITY
================

Current highest-priority work:

1. deterministic page classification
2. semantic section typing
3. retrieval routing
4. metadata filtering
5. canonical page scoring
6. weighted BM25 ranking
7. query expansion
8. retrieval explanations

====================================================
TARGET RETRIEVAL FLOW
=====================

Desired architecture:

```text id="jlwm51"
query
 ↓
intent detection
 ↓
metadata filtering
 ↓
candidate retrieval
 ↓
weighted BM25 ranking
 ↓
inspection/debugging
```

Later:

```text id="jlwm52"
query
 ↓
query expansion
 ↓
BM25
dense retrieval
 ↓
fusion
 ↓
reranker
 ↓
context builder
 ↓
LLM
```

====================================================
PAGE CLASSIFICATION GOAL
========================

Current classifier is unstable.

Need deterministic classification using:

* URL patterns
* title patterns
* heading patterns
* keyword voting

Allowed page types:

```python id="jlwm53"
[
  "Admissions",
  "Programs",
  "Faculty",
  "Placements",
  "Club",
  "Events",
  "Curriculum",
  "Research",
  "Facilities",
  "Notices",
  "Blog",
  "General"
]
```

====================================================
SEMANTIC SECTION TYPING
=======================

Each chunk should eventually include:

```json id="jlwm54"
{
  "section_type": "eligibility"
}
```

Allowed section types:

```python id="jlwm55"
[
  "eligibility",
  "fees",
  "placements",
  "faculty",
  "research",
  "facilities",
  "curriculum",
  "admissions",
  "hostel",
  "internships",
  "clubs",
  "events",
  "contact",
  "faq",
  "overview",
  "statistics",
  "syllabus",
  "general"
]
```

====================================================
RETRIEVAL PRIORITY
==================

Every chunk/document should receive:

```json id="jlwm56"
{
  "retrieval_priority": 0.0
}
```

Higher priority:

* admissions
* curriculum
* official program pages
* placements
* faculty

Lower priority:

* blogs
* notices
* event announcements
* duplicate content

====================================================
QUERY ROUTING
=============

We are moving toward intent-aware retrieval.

Example:

```text id="jlwm57"
"What is MCA eligibility?"
```

Should route to:

* Admissions pages
* eligibility sections

NOT:

* blogs
* events
* notices

====================================================
QUERY EXPANSION
===============

Institutional vocabulary is inconsistent.

Need lightweight synonym expansion.

Example:

```python id="jlwm58"
{
  "eligibility": [
    "criteria",
    "requirements",
    "admission criteria"
  ]
}
```

====================================================
RANKING STRATEGY
================

Planned ranking:

```python id="jlwm59"
final_score = (
    bm25_score * 0.7
    + retrieval_priority * 0.2
    + section_match_bonus * 0.1
)
```

Simple and inspectable.

====================================================
ENGINEERING PRINCIPLES
======================

IMPORTANT PROJECT RULES:

1. Prefer deterministic systems first
2. Avoid black-box retrieval
3. Retrieval explainability matters
4. Metadata quality matters more than model size
5. Chunk quality matters more than embeddings
6. Preserve factual information
7. Avoid aggressive cleaning
8. Prefer noisy-but-complete over clean-but-lossy
9. False-positive deletions are dangerous
10. Build evaluation before optimization

====================================================
WHAT NOT TO DO
==============

DO NOT:

* add LangChain-heavy architecture
* add agents prematurely
* add vector DB yet
* add embeddings yet
* use recursive character splitters
* build chatbot orchestration yet
* add graph RAG
* use LLM chunking
* over-abstract architecture
* aggressively summarize institutional content

====================================================
CURRENT TECHNOLOGY STACK
========================

Backend:

* Python 3.11+
* FastAPI

Core libraries:

* rank-bm25
* tiktoken
* spacy
* rapidfuzz
* regex
* orjson
* pydantic
* pytest

UI/debug:

* Streamlit

====================================================
FUTURE STACK (PLANNED)
======================

Embeddings:

```text id="jlwm60"
BAAI/bge-small-en-v1.5
```

Vector DB:
[Qdrant](https://qdrant.tech?utm_source=chatgpt.com)

Future retrieval:

* BM25
* dense retrieval
* reciprocal rank fusion
* reranking

Future reranker:

```text id="jlwm61"
bge-reranker-base
```

====================================================
LONG-TERM GOAL
==============

Final system should support:

* conversational RAG
* citations
* grounded answering
* admissions assistance
* faculty lookup
* curriculum navigation
* voice conversations
* streaming responses

Voice stack planned later:

* Pipecat
* streaming STT/TTS
* realtime retrieval

BUT NOT YET.

====================================================
CURRENT EXPECTATION FROM CLAUDE CODE
====================================

The coding agent should:

* continue existing architecture
* preserve deterministic design
* prioritize retrieval quality
* improve metadata consistency
* improve retrieval routing
* improve explainability
* improve evaluation

The coding agent should NOT:

* pivot architecture randomly
* introduce unnecessary frameworks
* add vector DB prematurely
* skip evaluation infrastructure
* overengineer abstractions

The project is already beyond prototype stage.
Now correctness, explainability, and retrieval precision matter most.
