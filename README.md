# Resume Tailor

A **dynamic resume builder** that generates a one-page, job-tailored PDF resume from a pasted job description. It uses RAG over a structured ChromaDB knowledge base and **Grok 4.3** (xAI) for section-by-section generation.

## Features

- Paste any job description and get a submission-ready one-page PDF
- **Section-isolated RAG** — each resume section pulls only its matching `doc_type` from Chroma, reducing cross-contamination (e.g. projects listed as job experience)
- **Embedding-ranked projects** — most relevant projects appear first; least relevant are trimmed if the PDF exceeds one page
- **Gradio UI** with multi-application threads and follow-up chat against your KB
- **GitHub integration** — distill public repo READMEs into resume-ready project summaries

## How it works

### Knowledge base (`data/`)

| Directory | `doc_type` | Contents |
|-----------|------------|----------|
| `header/` | `header` | Name, phone, email, GitHub |
| `education/` | `education` | Degrees and institutions |
| `jobs/` | `jobs` | Work and research experience |
| `skills/` | `skills` | Categorized skill lists |
| `projects/` | `projects` | Grok-distilled GitHub README summaries |
| `old_resumes/` | `resume` | Legacy PDF resumes (optional header fallback) |

The following templates were used with good results but can be modified to adjust user needs as needed.

#### header.md Template
```
# Header
1. name: your name
2. phone number: your phone number
3. email: your email
4. github: your github
```

#### education.md Template
```
# EDUCATION:

Grad Degree Title, Institution Name — Date Earned/Expected

Undergrad Degree Title, Institution Name — Date Earned/Expected
```

#### job.md Template
```
# Company Name, City, State Month Year - Month Year
1. Title: your title
2. Location: City, State
3. Dates: Month Year - Month Year

## Details
1. First achievment description
   
2. Second achievement description

...
```
Note: Each job should have it's own .md in the jobs directory

#### skills.md Template
```
# SKILLS

## Programming Languages: 
your languages

## AI/ML Frameworks:
your frameworks 

## Tools/Platforms:
your tools/platforms 

## Concepts:
your concepts/knowledge

```

### Ingestion

`ingest.py` embeds all `data/` files into ChromaDB (`chroma_db/`, collection `rag_collection`) using **local Ollama embeddings** (`nomic-embed-text`). Re-running ingest wipes and rebuilds the database.

`ingest_github.py` fetches READMEs from a GitHub user's public repos, distills them with Grok 4.3, and writes summaries to `data/projects/`.

### Generation (`app.py`)

On **Generate Tailored Resume**:

1. User pastes a job description.
2. The description is embedded; Chroma is queried **per section** using `doc_type` metadata filters.
3. **Grok 4.3** (xAI API, OpenAI-compatible client) generates each section in isolation.
4. **Section order:** Header (`header` → `resume` fallback) → Experience (`jobs`) → Projects (`projects`, embedding-ranked) → Skills (`skills`) → Education (`education`) → Objective (job description + generated Experience/Projects/Skills).
5. Sections merge into JSON; projects are re-sorted by embedding relevance; `resume_pdf.py` renders the PDF.
6. If the output exceeds one page, `fit_resume_to_one_page()` drops the least relevant project(s) and rebuilds until ≤1 page.
7. User downloads the PDF; chat supports follow-up Q&A against the KB.

## Prerequisites

- **Python 3.13+** 
- **[Ollama](https://ollama.com/)** running locally with `nomic-embed-text` pulled
- **xAI API key** for Grok 4.3 generation (`XAI_API_KEY`)
- Optional: **GitHub token** for higher-rate `ingest_github.py` API calls

## Setup

```bash
# Clone and enter the project
cd rag-portfolio

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
XAI_API_KEY=your-xai-api-key
GITHUB_OWNER=your-github-username
GITHUB_TOKEN=your-github-token
```

Pull the embedding model (requires Ollama running):

```bash
ollama pull nomic-embed-text
```

## Usage

```bash
# 1. (Optional) Refresh project summaries from GitHub
python ingest_github.py

# 2. Build / rebuild the Chroma knowledge base
python ingest.py

# 3. Launch the Gradio UI
python app.py
```

Open the URL shown in the terminal (default `http://127.0.0.1:7860`), paste a job description, and click **Generate Tailored Resume**. PDFs are saved to `generated_resumes/`.

## Project structure

```
rag-portfolio/
├── app.py              # Gradio UI, section-by-section resume generation
├── resume_pdf.py       # ReportLab PDF renderer and one-page fit logic
├── ingest.py           # Embed data/ into ChromaDB
├── ingest_github.py    # GitHub README → Grok summaries → data/projects/
├── data/               # Knowledge base source files
├── chroma_db/          # Persisted vector store (generated)
├── generated_resumes/  # Output PDFs (generated)
├── grok_workspace/     # Agent state, changelog, summaries
├── AGENTS.md           # Grok project rules for contributors/agents
└── requirements.txt
```

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `XAI_API_KEY` | Yes | Grok 4.3 generation via xAI API |
| `GITHUB_OWNER` | No | GitHub username for `ingest_github.py` (default: `ajbax-cmd`) |
| `GITHUB_TOKEN` | No | GitHub API auth for higher rate limits |

## Tech stack

- **LLM:** Grok 4.3 via xAI OpenAI-compatible API
- **Embeddings:** Ollama `nomic-embed-text` (local)
- **Vector store:** ChromaDB
- **UI:** Gradio
- **PDF:** ReportLab + pypdf (page counting)

## For contributors / AI agents

See `AGENTS.md` for project conventions and `grok_workspace/REPO_STATE.md` for current architecture notes and continuity context.

## Disclaimer

Resume quality depends entirely on the accuracy of your knowledge base. You are solely responsible for the information you add to it. Use this tool responsibly — the owner does not condone fabricated credentials or experience and accepts no liability for resumes built from false or misleading KB content.