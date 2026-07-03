# RAG-PORTFOLIO PROJECT -> DYNAMIC RESUME BUILDER
- REPO_STATE.md path: grok_workspace/REPO_STATE.md

- CHANGELOG.md path: grok_workspace/CHANGELOG.md

- Summaries directory path: grok_workspace/summaries

Read REPO_STATE.md if you need more project context.

## Overview
This project is a **dynamic resume builder** that tailors a one-page PDF resume to a pasted job description, using a user-maintained knowledge base (KB) and RAG over ChromaDB.

**KB sources (`data/`):**
- `header/` — name, phone, email, GitHub (`doc_type: header`)
- `education/` — degrees and institutions (`doc_type: education`)
- `jobs/` — work and research experience (`doc_type: jobs`)
- `skills/` — categorized skill lists (`doc_type: skills`)
- `projects/` — Grok-distilled GitHub README summaries (`doc_type: projects`; produced by `ingest_github.py`)
- `old_resumes/` — legacy PDF resumes, optional fallback (`doc_type: resume`)

**Ingestion:** `ingest.py` loads the above into ChromaDB (`chroma_db/`, collection `rag_collection`) using **local Ollama embeddings** (`nomic-embed-text`). Re-running `ingest.py` wipes and rebuilds the DB.

**Generation:** `app.py` is a Gradio UI. On "Generate Tailored Resume":
1. User pastes a job description.
2. The job description is embedded; Chroma is queried **per section** using `doc_type` metadata filters (not one monolithic prompt).
3. **Grok 4.3** (xAI API, OpenAI-compatible client) generates each section in isolation to reduce cross-contamination (e.g. projects listed as job achievements).
4. Section order: **Header** (`header` → `resume` fallback) → **Experience** (`jobs`) → **Projects** (`projects`, embedding-ranked) → **Skills** (`skills`) → **Education** (`education`) → **Objective** (job description + generated Experience/Projects/Skills).
5. Sections are merged into JSON, projects re-sorted by embedding relevance, then `resume_pdf.py` renders a PDF. If output exceeds one page, `fit_resume_to_one_page()` drops the **least relevant** project(s) and rebuilds until ≤1 page.
6. User downloads the PDF; a lightweight chat supports follow-up Q&A against the KB.

**Key env vars:** `XAI_API_KEY` (required for generation); `GITHUB_TOKEN` / `GITHUB_OWNER` (optional, for `ingest_github.py`).

## Instructions
For every prompt that initiates changes to the code base:

1. Update the state of REPO_STATE.md with a last updated timestamp.
2. Write a summary called <Relevant_Title_of_Changes>_summary.md in the summaries directory that uses <>_summary.md format
3. Update CHANGELOG.md. Create if it does not exist.

### <Relevant_Title_of_Changes>_summary.md Format


Prompt:

---
[Copy the user's prompt that initiated the code change exactly]
---

Summary:
[Summarize what changes were made]

Files Effected:
[
- file1.py
    - function1: what was done
    - function2: what was done
    - global (no function): what was done
- file2.py
    - function1 : what was done
]

### CHANGELOG.md Format
- YYYY-MM-DD [Relveant Link Name](<Relevant_Title_of_Changes>_summary.md)

