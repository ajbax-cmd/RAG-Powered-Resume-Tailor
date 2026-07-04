"""Render structured resume data as a one-page PDF."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from pypdf import PdfReader

OUTPUT_DIR = Path(__file__).resolve().parent / "generated_resumes"


def count_pdf_pages(pdf_path: str) -> int:
    return len(PdfReader(pdf_path).pages)


def fit_resume_to_one_page(
    resume_data: dict,
    job_title: str = "",
    *,
    min_projects: int = 2,
) -> tuple[str, dict, int]:
    """Build PDF, dropping least-relevant projects (last in list) until one page."""
    data = dict(resume_data)
    projects = list(data.get("projects", []))
    data["projects"] = projects

    pdf_path = build_resume_pdf(data, job_title=job_title)
    pages = count_pdf_pages(pdf_path)

    while pages > 1 and len(projects) > min_projects:
        projects.pop()
        data["projects"] = projects
        pdf_path = build_resume_pdf(data, job_title=job_title)
        pages = count_pdf_pages(pdf_path)

    return pdf_path, data, pages


def _xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _education_lines(entry: str | dict) -> list[str]:
    """Return ReportLab paragraph markup lines for one education entry."""
    if isinstance(entry, dict):
        degree = _xml_escape(entry.get("degree", "").strip())
        institution = _xml_escape(entry.get("institution", "").strip())
        dates = _xml_escape(entry.get("dates", "").strip())
        lines: list[str] = []
        if degree and dates:
            lines.append(f"<b>{degree}</b>  {dates}")
        elif degree:
            lines.append(f"<b>{degree}</b>")
        elif dates:
            lines.append(dates)
        if institution:
            lines.append(institution)
        return lines

    text = str(entry).strip()
    if not text:
        return []

    # String fallback: "Degree, Institution, Date" or "Degree, Institution: Date"
    if ":" in text:
        left, dates = text.rsplit(":", 1)
        dates = dates.strip()
        if "," in left:
            degree, institution = left.split(",", 1)
            degree = _xml_escape(degree.strip())
            institution = _xml_escape(institution.strip())
            dates = _xml_escape(dates)
            return [f"<b>{degree}</b>  {dates}", institution]

    if "," in text:
        degree, rest = text.split(",", 1)
        degree = _xml_escape(degree.strip())
        rest = _xml_escape(rest.strip())
        return [f"<b>{degree}</b>, {rest}"]

    return [f"<b>{_xml_escape(text)}</b>"]


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    return cleaned[:40] or "resume"


def build_resume_pdf(resume_data: dict, job_title: str = "") -> str:
    """Build a one-page resume PDF and return the file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_filename(job_title or resume_data.get("name", "resume"))
    output_path = OUTPUT_DIR / f"resume_{slug}_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "Name",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=2,
        textColor=colors.HexColor("#1a1a2e"),
    )
    contact_style = ParagraphStyle(
        "Contact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        alignment=TA_CENTER,
        spaceAfter=8,
        textColor=colors.HexColor("#4a4a68"),
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=11,
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        spaceAfter=2,
        textColor=colors.HexColor("#222222"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=1,
    )
    role_style = ParagraphStyle(
        "Role",
        parent=body_style,
        fontName="Helvetica-Bold",
        spaceAfter=1,
    )

    story: list = []

    name = resume_data.get("name", "").strip()
    if name:
        story.append(Paragraph(name.upper(), name_style))

    contact = resume_data.get("contact", "").strip()
    if contact:
        story.append(Paragraph(contact, contact_style))

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.75,
            color=colors.HexColor("#3d5a80"),
            spaceBefore=0,
            spaceAfter=4,
        )
    )

    def add_section(title: str) -> None:
        story.append(Paragraph(title.upper(), section_style))

    summary = resume_data.get("summary", "").strip()
    if summary:
        add_section("Summary")
        story.append(Paragraph(summary, body_style))

    education = resume_data.get("education", [])
    if education:
        add_section("Education")
        for entry in education:
            lines = _education_lines(entry)
            for line in lines:
                story.append(Paragraph(line, body_style))

    skills = resume_data.get("skills", [])
    if skills:
        add_section("Skills")
        if isinstance(skills, str):
            skills = [line.strip() for line in re.split(r"[\n;]+", skills) if line.strip()]
        for skill in skills:
            text = str(skill).strip().lstrip("•").lstrip("-").strip()
            if text:
                story.append(Paragraph(f"• {text}", bullet_style))

    experience = resume_data.get("experience", [])
    if experience:
        add_section("Experience")
        for item in experience:
            header_parts = [
                part
                for part in [
                    item.get("title", ""),
                    item.get("company", ""),
                    item.get("dates", ""),
                ]
                if part
            ]
            if header_parts:
                story.append(Paragraph(" — ".join(header_parts[:2]) + (f"  {header_parts[2]}" if len(header_parts) > 2 else ""), role_style))
            for bullet in item.get("bullets", []):
                story.append(Paragraph(f"• {bullet}", bullet_style))

    projects = resume_data.get("projects", [])
    if projects:
        add_section("Projects")
        for item in projects:
            title = item.get("title", "").strip()
            if title:
                story.append(Paragraph(title, role_style))
            for bullet in item.get("bullets", []):
                story.append(Paragraph(f"• {bullet}", bullet_style))

    doc.build(story)
    return str(output_path)