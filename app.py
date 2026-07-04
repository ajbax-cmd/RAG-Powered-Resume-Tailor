import json
import os
import re
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI

from resume_pdf import build_resume_pdf, fit_resume_to_one_page

# Configuration
LLM_MODEL = "grok-4.3"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

CLIENT = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)



SECTION_BASE_RULES = """You are an expert resume writer. Use ONLY the provided context for this section.
Do not invent employers, degrees, dates, projects, or skills not supported by the context.
Write achievement-oriented bullets (15-28 words) with metrics when available.
Return ONLY valid JSON with no markdown fences or extra commentary."""

EXPERIENCE_SYSTEM_PROMPT = SECTION_BASE_RULES + """

Generate ONLY the Experience section for a job-tailored resume.
- Include every job and research role found in the context.
- Tailor bullet emphasis to match the job description.
- Provide 3-4 bullets per role.

Return JSON:
{{"experience": [{{"title": "...", "company": "...", "dates": "...", "bullets": ["..."]}}]}}"""

PROJECTS_SYSTEM_PROMPT = SECTION_BASE_RULES + """

Generate ONLY the Projects section for a job-tailored resume.
- Include 4-6 projects from the context, ranked most-to-least relevant to the job.
- Preserve the order of project summaries in the context (highest embedding match first).
- Provide 2-3 bullets per project.

Return JSON:
{{"projects": [{{"title": "Project Name — descriptor", "bullets": ["..."]}}]}}"""

SKILLS_SYSTEM_PROMPT = SECTION_BASE_RULES + """

Generate ONLY the Skills section for a job-tailored resume.
- Prioritize skills that match the job description.
- Use 4-6 categorized lines (e.g. "Languages: ...", "ML/AI: ...", "Tools: ...").

Return JSON:
{{"skills": ["Languages: ...", "ML/AI: ...", "Tools: ..."]}}"""

EDUCATION_SYSTEM_PROMPT = SECTION_BASE_RULES + """

Generate ONLY the Education section. This section is usually stable — use exact facts from context.
- Provide 2 structured entries (one per degree).
- Do not include GPAs unless explicitly present in the context.

Return JSON:
{{"education": [{{"degree": "...", "institution": "...", "dates": "..."}}]}}"""

SUMMARY_SYSTEM_PROMPT = SECTION_BASE_RULES + """
Generate ONLY the Summary section for a job-tailored resume.

FRAMING:
- Write from the employer's perspective — what the candidate brings, not what they want.
- Never begin with "Seeking", "Looking to", "I am", or "Eager to".
- Avoid filler phrases: "passionate about", "eager to contribute", "proven track record", "looking to leverage", "results-driven".

STRUCTURE (exactly 2 sentences):
- Sentence 1: Candidate's core technical identity and experience level.
- Sentence 2: Most relevant experience or differentiator tied directly to the target role.

SYNTHESIS RULES:
- Reference the target role and highlight strengths from the already-generated sections provided.
- Prioritize Experience and Skills sections over Projects when selecting what to highlight.
- Mirror 2-3 keywords from the job description naturally — do not force them.
- Be specific: prefer "3-year ML pipeline developer" over "experienced software engineer".
- Length: 2-3 lines when rendered. No run-on sentences.

Return JSON:
{{"summary": "2 sentence tailored summary"}}"""

HEADER_SYSTEM_PROMPT = SECTION_BASE_RULES + """

Extract the candidate header from the resume context.
Return JSON:
{{"name": "Full Name", "contact": "phone || email || linkedin/github URL"}}"""

CHAT_SYSTEM_PROMPT = """You are a helpful resume and career assistant with access to the user's knowledge base (past resumes, projects, and experience).

Use the provided context to answer questions accurately. If information is not in the context, say so clearly.
When discussing resume strategy, be specific and actionable."""


def _format_docs(docs: list, label: str = "") -> str:
    if not docs:
        return ""
    header = f"=== {label} ===\n" if label else ""
    body = "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')} ({d.metadata.get('doc_type', 'resume')})\n{d.page_content}"
        for d in docs
    )
    return header + body


def _dedupe_by_source(docs: list, limit: int | None = None) -> list:
    seen: set[str] = set()
    unique: list = []
    for doc in docs:
        source = doc.metadata.get("source", doc.page_content[:80])
        if source in seen:
            continue
        seen.add(source)
        unique.append(doc)
        if limit and len(unique) >= limit:
            break
    return unique


def get_ranked_docs(
    query: str,
    doc_type: str,
    k: int = 12,
) -> list[tuple[object, float]]:
    """Documents of a given doc_type, ordered by embedding similarity (best first)."""
    ranked = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter={"doc_type": doc_type},
    )
    deduped: list[tuple[object, float]] = []
    seen: set[str] = set()
    for doc, score in ranked:
        source = doc.metadata.get("source", doc.page_content[:80])
        if source in seen:
            continue
        seen.add(source)
        deduped.append((doc, score))
    return deduped


def get_ranked_project_docs(job_description: str) -> list[tuple[object, float]]:
    """Project chunks ordered by embedding similarity to the job description (best first)."""
    return get_ranked_docs(job_description, "projects", k=12)


def get_section_context(
    query: str,
    doc_type: str,
    *,
    k: int = 12,
    label: str = "",
    ranked: list[tuple[object, float]] | None = None,
) -> str:
    if ranked is None:
        ranked = get_ranked_docs(query, doc_type, k=k)
    docs = [doc for doc, _ in ranked]
    return _format_docs(docs, label)


def _project_match_keys(source: str) -> list[str]:
    stem = Path(source).stem.lower()
    if stem.startswith("readme_"):
        stem = stem[len("readme_") :]
    keys = [stem.replace("-", " ").replace("_", " ")]
    keys.append(stem.replace("-", "").replace("_", ""))
    keys.extend(part for part in re.split(r"[-_\s]+", stem) if len(part) > 2)
    return keys


def order_resume_projects(
    resume_data: dict,
    ranked_projects: list[tuple[object, float]],
) -> dict:
    """Re-sort LLM project output to match embedding relevance (most relevant first)."""
    rank_by_source = {
        doc.metadata.get("source", ""): (index, score)
        for index, (doc, score) in enumerate(ranked_projects)
    }
    source_keys = {
        source: _project_match_keys(source)
        for source in rank_by_source
    }

    def project_sort_key(project: dict) -> tuple[int, float]:
        title = project.get("title", "").lower()
        best_rank = len(ranked_projects)
        best_score = float("inf")
        for source, (rank, score) in rank_by_source.items():
            if any(key in title for key in source_keys.get(source, [])):
                if rank < best_rank or (rank == best_rank and score < best_score):
                    best_rank = rank
                    best_score = score
        return (best_rank, best_score)

    projects = list(resume_data.get("projects", []))
    resume_data["projects"] = sorted(projects, key=project_sort_key)
    return resume_data


def get_local_context(question: str) -> str:
    docs = _dedupe_by_source(vectorstore.similarity_search(question, k=10), limit=8)
    return _format_docs(docs)


def count_resume_words(resume_data: dict) -> int:
    parts: list[str] = []
    for key in ("name", "contact", "summary"):
        value = resume_data.get(key, "")
        if isinstance(value, str):
            parts.append(value)
    for key in ("education", "skills"):
        for item in resume_data.get(key, []):
            parts.append(str(item))
    for section in ("experience", "projects"):
        for item in resume_data.get(section, []):
            for field in ("title", "company", "dates"):
                parts.append(str(item.get(field, "")))
            for bullet in item.get("bullets", []):
                parts.append(str(bullet))
    return len(" ".join(parts).split())


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def thread_label(thread_id: str) -> str:
    return f"Application {thread_id}"


def label_to_id(label: str) -> str:
    return label.rsplit(" ", 1)[-1]


def thread_choices(threads: dict[str, list]) -> list[str]:
    return [thread_label(tid) for tid in sorted(threads, key=int)]


def format_chat_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for message in history[-8:]:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_json(text: str) -> dict | list:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1:
                text = text[start : end + 1]
                break
    return json.loads(text)


def extract_section(text: str, section_key: str):
    parsed = extract_json(text)
    if isinstance(parsed, dict) and section_key in parsed:
        return parsed[section_key]
    if section_key in ("skills", "education", "experience", "projects") and isinstance(parsed, list):
        return parsed
    if section_key == "summary" and isinstance(parsed, str):
        return parsed
    raise ValueError(f"Could not extract '{section_key}' from model response")


def call_grok(system_prompt: str, user_prompt: str, *, temperature: float = 0.3) -> str:
    response = CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        reasoning_effort="high",
        max_completion_tokens=4000,
    )
    return response.choices[0].message.content


def _section_user_prompt(
    job_description: str,
    context: str,
    *,
    section_name: str,
    extra: str = "",
    expand: bool = False,
) -> str:
    prompt = f"""Job Description:
{job_description}

{section_name} Context (use ONLY this context for this section):
{context}
{extra}"""
    if expand:
        prompt += """

The previous draft for this section was too short. Add more detail from the context above.
Do not remove existing facts — only expand bullets and entries."""
    return prompt


def generate_header_section(job_description: str) -> dict[str, str]:
    context = get_section_context(
        job_description,
        "header",
        k=4,
        label="Header",
    )
    if not context:
        context = get_section_context(
            job_description,
            "resume",
            k=4,
            label="Resume Header (fallback)",
        )
    raw = call_grok(
        HEADER_SYSTEM_PROMPT,
        _section_user_prompt(job_description, context, section_name="Header"),
    )
    parsed = extract_json(raw)
    return {
        "name": parsed.get("name", ""),
        "contact": parsed.get("contact", ""),
    }


def generate_experience_section(job_description: str, *, expand: bool = False) -> list:
    context = get_section_context(
        job_description,
        "jobs",
        k=8,
        label="Job Experience",
    )
    raw = call_grok(
        EXPERIENCE_SYSTEM_PROMPT,
        _section_user_prompt(
            job_description,
            context,
            section_name="Experience",
            expand=expand,
        ),
        temperature=0.35 if expand else 0.3,
    )
    return extract_section(raw, "experience")


def generate_projects_section(
    job_description: str,
    ranked_projects: list[tuple[object, float]],
    *,
    expand: bool = False,
) -> list:
    context = get_section_context(
        job_description,
        "projects",
        label="Projects (most-to-least relevant)",
        ranked=ranked_projects,
    )
    raw = call_grok(
        PROJECTS_SYSTEM_PROMPT,
        _section_user_prompt(
            job_description,
            context,
            section_name="Projects",
            extra="\nOrder projects from most relevant to least relevant, matching the context order.",
            expand=expand,
        ),
        temperature=0.35 if expand else 0.3,
    )
    return extract_section(raw, "projects")


def generate_skills_section(job_description: str) -> list:
    context = get_section_context(
        job_description,
        "skills",
        k=6,
        label="Skills",
    )
    raw = call_grok(
        SKILLS_SYSTEM_PROMPT,
        _section_user_prompt(job_description, context, section_name="Skills"),
    )
    return extract_section(raw, "skills")


def generate_education_section(job_description: str) -> list:
    context = get_section_context(
        job_description,
        "education",
        k=4,
        label="Education",
    )
    raw = call_grok(
        EDUCATION_SYSTEM_PROMPT,
        _section_user_prompt(job_description, context, section_name="Education"),
    )
    return extract_section(raw, "education")


def _format_section_summary(section_name: str, content) -> str:
    return f"{section_name}:\n{json.dumps(content, indent=2)}"


def generate_summary_section(
    job_description: str,
    *,
    experience: list,
    projects: list,
    skills: list,
) -> str:
    generated_context = "\n\n".join(
        [
            _format_section_summary("Experience", experience),
            _format_section_summary("Projects", projects),
            _format_section_summary("Skills", skills),
        ]
    )
    raw = call_grok(
        SUMMARY_SYSTEM_PROMPT,
        f"""Job Description:
{job_description}

Already-generated resume sections (use for tailoring the summary):
{generated_context}""",
    )
    return extract_section(raw, "summary")


def build_resume_by_sections(
    job_description: str,
    *,
    expand: bool = False,
) -> tuple[dict, list[tuple[object, float]]]:
    """Assemble a full resume JSON via isolated section-by-section generation."""
    ranked_projects = get_ranked_project_docs(job_description)

    header = generate_header_section(job_description)
    experience = generate_experience_section(job_description, expand=expand)
    projects = generate_projects_section(job_description, ranked_projects, expand=expand)
    skills = generate_skills_section(job_description)
    education = generate_education_section(job_description)
    summary = generate_summary_section(
        job_description,
        experience=experience,
        projects=projects,
        skills=skills,
    )

    resume_data = {
        "name": header.get("name", ""),
        "contact": header.get("contact", ""),
        "summary": summary,
        "education": education,
        "skills": skills,
        "experience": experience,
        "projects": projects,
    }
    return resume_data, ranked_projects


def generate_resume(
    job_description: str,
    history: list[dict],
    active_thread: str,
    threads: dict[str, list],
):
    if not job_description or not job_description.strip():
        return history, None, threads, "Please paste a job description to generate a tailored resume."

    job_description = job_description.strip()

    try:
        resume_data, ranked_projects = build_resume_by_sections(job_description)
        word_count = count_resume_words(resume_data)
        project_count = len(resume_data.get("projects", []))

        if word_count < 400 or project_count < 3:
            resume_data, ranked_projects = build_resume_by_sections(
                job_description,
                expand=True,
            )

        resume_data = order_resume_projects(resume_data, ranked_projects)
        pdf_path, resume_data, page_count = fit_resume_to_one_page(
            resume_data,
            job_title=job_description[:80],
        )

        trimmed = project_count - len(resume_data.get("projects", []))
        trim_note = (
            f"\n**Trimmed:** removed {trimmed} least-relevant project(s) to fit one page."
            if trimmed > 0
            else ""
        )

        summary = (
            f"**Resume generated** for this role.\n\n"
            f"**Candidate:** {resume_data.get('name', 'N/A')}\n"
            f"**Projects:** {len(resume_data.get('projects', []))} (ordered by relevance)\n"
            f"**Pages:** {page_count}{trim_note}\n"
            f"**PDF ready** — download below."
        )

        history = history + [
            {"role": "user", "content": f"Generate resume for:\n\n{job_description[:500]}..."},
            {"role": "assistant", "content": summary},
        ]

        threads = dict(threads)
        threads[active_thread] = history
        get_session_history(active_thread).add_user_message(job_description)
        get_session_history(active_thread).add_ai_message(summary)

        return history, pdf_path, threads, ""

    except json.JSONDecodeError:
        return (
            history,
            None,
            threads,
            "The model returned invalid JSON. Please try again.",
        )
    except Exception as exc:
        return history, None, threads, f"Error generating resume: {exc}"


def chat_with_kb(message: str, session_id: str = "default") -> str:
    context = get_local_context(message)
    history = get_session_history(session_id).messages
    history_text = "\n".join(
        f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', '')}"
        for m in history[-6:]
    )

    user_prompt = f"""Context from knowledge base:
{context}

Recent chat:
{history_text or "No previous messages."}

User question: {message}"""

    response = call_grok(CHAT_SYSTEM_PROMPT, user_prompt)
    get_session_history(session_id).add_user_message(message)
    get_session_history(session_id).add_ai_message(response)
    return response


def respond(message: str, history: list[dict], active_thread: str, threads: dict[str, list]):
    if not message or not message.strip():
        return "", history, threads

    bot_message = chat_with_kb(message, session_id=active_thread)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": str(bot_message)},
    ]

    threads = dict(threads)
    threads[active_thread] = history
    return "", history, threads


def select_thread(selected: str, active_thread: str, threads: dict[str, list], history: list[dict]):
    if not selected:
        return active_thread, history, threads, None

    threads = dict(threads)
    threads[active_thread] = history

    new_thread = label_to_id(selected)
    loaded_history = threads.get(new_thread, [])
    return new_thread, loaded_history, threads, None


def new_chat(active_thread: str, threads: dict[str, list], history: list[dict]):
    threads = dict(threads)
    threads[active_thread] = history

    new_thread = str(max((int(tid) for tid in threads), default=0) + 1)
    threads[new_thread] = []

    choices = thread_choices(threads)
    return (
        new_thread,
        threads,
        [],
        "",
        "",
        None,
        gr.Dropdown(choices=choices, value=thread_label(new_thread)),
    )


def clear_chat(active_thread: str, threads: dict[str, list]):
    threads = dict(threads)
    threads[active_thread] = []
    if active_thread in store:
        store[active_thread].clear()
    return [], threads, None


# Load vectorstore
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

store: dict[str, BaseChatMessageHistory] = {}

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
.hero-title {
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-subtitle {
    text-align: center;
    color: #5c6b7a;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}
.panel-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    background: #fafbfc;
}
.pdf-panel {
    border: 2px dashed #3d5a80;
    border-radius: 12px;
    padding: 1.25rem;
    background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    text-align: center;
}
"""

theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_title_text_weight="600",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(title="Resume Tailor") as demo:
    gr.Markdown(
        """
        <div class="hero-title">
        <h1>Resume Tailor</h1>
        </div>
        <div class="hero-subtitle">
        RAG-powered, job-specific resumes from your Chroma knowledge base · Powered by Grok 4.3
        </div>
        """,
    )

    active_thread = gr.State("1")
    threads = gr.State({"1": []})

    with gr.Row():
        thread_dropdown = gr.Dropdown(
            choices=["Application 1"],
            value="Application 1",
            label="Application thread",
            scale=3,
        )
        new_chat_btn = gr.Button("New Application", variant="secondary", scale=1)

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.Markdown("### Job Description")
            job_description = gr.Textbox(
                placeholder="Paste the full job posting here — title, requirements, responsibilities...",
                label="Target role",
                lines=14,
                max_lines=20,
            )
            generate_btn = gr.Button("Generate Tailored Resume", variant="primary", size="lg")
            status_msg = gr.Markdown("")

        with gr.Column(scale=3):
            gr.Markdown("### Conversation")
            chatbot = gr.Chatbot(
                height=420,
                label="Chat",
            )
            msg = gr.Textbox(
                placeholder="Ask about your experience, projects, or resume strategy...",
                label="Follow-up question",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="secondary")
                clear_btn = gr.Button("Clear Chat")

    gr.Markdown("### Your Resume")
    with gr.Group(elem_classes=["pdf-panel"]):
        resume_pdf = gr.File(
            label="Download one-page PDF resume",
            file_types=[".pdf"],
            interactive=False,
        )
        gr.Markdown(
            "*A formatted, one-page PDF will appear here after you generate a resume.*",
        )

    generate_btn.click(
        generate_resume,
        [job_description, chatbot, active_thread, threads],
        [chatbot, resume_pdf, threads, status_msg],
    )

    submit_btn.click(
        respond,
        [msg, chatbot, active_thread, threads],
        [msg, chatbot, threads],
    )
    msg.submit(
        respond,
        [msg, chatbot, active_thread, threads],
        [msg, chatbot, threads],
    )

    thread_dropdown.change(
        select_thread,
        [thread_dropdown, active_thread, threads, chatbot],
        [active_thread, chatbot, threads, resume_pdf],
    )

    new_chat_btn.click(
        new_chat,
        [active_thread, threads, chatbot],
        [active_thread, threads, chatbot, msg, job_description, resume_pdf, thread_dropdown],
    )

    clear_btn.click(
        clear_chat,
        [active_thread, threads],
        [chatbot, threads, resume_pdf],
    )

if __name__ == "__main__":
    print("Starting Resume Tailor...")
    print(f"LLM: {LLM_MODEL} (xAI)")
    print(f"Embeddings: {EMBEDDING_MODEL} (Ollama)")
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )