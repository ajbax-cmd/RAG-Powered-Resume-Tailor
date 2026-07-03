# REPO_STATE

**Last updated:** 2026-06-28 (README + requirements.txt)

## What this repo is

RAG-powered resume tailor: paste a job description → get a one-page tailored PDF resume built from a structured Chroma knowledge base. Gradio UI (`app.py`), local embeddings (Ollama), generation via Grok 4.3 (xAI).

## Architecture (quick)

```
data/  ──► ingest.py (Ollama embed) ──► chroma_db/
                                              │
ingest_github.py ──► data/projects/ ──────────┘
                                              │
job description ──► app.py (section RAG + Grok) ──► resume_pdf.py ──► generated_resumes/*.pdf
```

## Key files

| File | Role |
|------|------|
| `app.py` | Gradio UI, section-by-section resume generation, chat, thread management |
| `resume_pdf.py` | ReportLab PDF renderer, page count, one-page fit loop |
| `ingest.py` | Ingest `data/` into Chroma with `doc_type` metadata |
| `ingest_github.py` | Fetch GitHub READMEs → Grok-distilled project summaries → `data/projects/` |
| `AGENTS.MD` | Agent instructions for code-change workflow (summaries, changelog) |
| `.env` | `XAI_API_KEY`, optional `GITHUB_TOKEN`, `GITHUB_OWNER` |

## Data layout (`data/`)

| Path | `doc_type` | Notes |
|------|------------|-------|
| `header/` | `header` | `header.md` — name, phone, email, GitHub |
| `education/` | `education` | `EDUCATION.md` |
| `jobs/` | `jobs` | One file per role (e.g. `Penn State ARL.md`, `Research Assistant.md`) |
| `skills/` | `skills` | `SKILLS.md` — categorized bullets |
| `projects/` | `projects` | `readme_<repo>.txt` from `ingest_github.py` |
| `old_resumes/` | `resume` | Legacy PDFs; header fallback only |

## Chroma / RAG conventions

- **DB path:** `chroma_db/`
- **Collection:** `rag_collection`
- **Embeddings:** `nomic-embed-text` via Ollama (`OllamaEmbeddings`)
- **Retrieval helpers in `app.py`:**
  - `get_ranked_docs(query, doc_type, k)` — similarity search + dedupe by source
  - `get_section_context(...)` — formats isolated context for one section
  - `order_resume_projects()` — re-sorts LLM project output to match embedding rank

## Section generation flow (`build_resume_by_sections`)

1. `generate_header_section` — `doc_type: header`, fallback `resume`
2. `generate_experience_section` — `doc_type: jobs`
3. `generate_projects_section` — `doc_type: projects` (pre-ranked)
4. `generate_skills_section` — `doc_type: skills`
5. `generate_education_section` — `doc_type: education`
6. `generate_objective_section` — uses job description + already-generated Experience, Projects, Skills (no extra Chroma query)

**Expand retry:** If total word count &lt; 400 or &lt; 3 projects, re-run experience + projects with `expand=True`.

**PDF trim:** `fit_resume_to_one_page()` in `resume_pdf.py` — if pages &gt; 1, pop `projects[-1]` (least relevant) until ≤1 page (min 2 projects).

## PDF formatting notes (`resume_pdf.py`)

- Education: degree **bold**, institution/dates regular; supports structured dict or string entries
- Skills: bulleted list (one `•` line per category)
- Experience/projects: role title bold line + bullet points
- Output dir: `generated_resumes/`

## How to run

```bash
# Prerequisites: Ollama running with nomic-embed-text pulled
python ingest_github.py   # optional: refresh project summaries from GitHub
python ingest.py          # rebuild Chroma (deletes existing chroma_db/)
python app.py             # Gradio at http://127.0.0.1:7860
```

## LLM / API config

- **Generation model:** `grok-4.3` via `openai.OpenAI(base_url="https://api.x.ai/v1")`
- **Embeddings:** local Ollama only (not xAI)
- `ingest_github.py` docstring still mentions Gemini; implementation uses Grok 4.3

## Gradio UI features

- Multi-thread "Application" dropdown (separate job applications)
- Job description panel + generate button
- Chat for KB Q&A (`chat_with_kb`, unfiltered retrieval)
- PDF download panel

## Known quirks / gotchas

- `ingest.py` always `shutil.rmtree(CHROMA_DIR)` — full rebuild every run
- `doc_type` for jobs is `"jobs"` (plural), projects is `"projects"` (plural) — filters must match exactly
- Header generation requires `header` chunks in Chroma; re-ingest after editing `data/header/header.md`
- `requirements.txt` is minimal; actual deps live in `.venv` (gradio, langchain-*, reportlab, pypdf, etc.)
- Agent instructions file is `AGENTS.MD` (uppercase extension on disk)

## Recent design decisions (for continuity)

- Moved from one-shot full-resume JSON to **section-by-section** prompts to reduce hallucination/cross-contamination
- Projects ordered by **embedding similarity** to job description; trim loop removes least relevant last
- Objective generated **last**, using synthesized content from other sections (employer-framed, no "Seeking..." openers)

## Suggested next work (not implemented)

- Root-level `REPO_STATE.md` symlink or note pointing to `grok_workspace/REPO_STATE.md`
- Populate `grok_workspace/summaries/` as changes land (per `AGENTS.MD`)
- Optional: skip full Chroma wipe in `ingest.py` for incremental updates