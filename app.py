import io
import json
import os
import re
from typing import Any

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


APP_TITLE = "AI Resume ATS Analyzer"
MODEL_NAME = "gemini-2.5-flash"

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_RESUME_CHARS = 50000
MAX_JOB_DESCRIPTION_CHARS = 30000
MIN_RESUME_CHARS = 100

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        }
        .score-card {
            padding: 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            text-align: center;
            margin-bottom: 1rem;
        }
        .score-number {
            font-size: 3.5rem;
            font-weight: 800;
            line-height: 1;
        }
        .score-label {
            font-size: 0.95rem;
            color: #666;
            margin-top: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read GEMINI_API_KEY from Streamlit Secrets or environment."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    env_key = os.getenv("GEMINI_API_KEY")
    return env_key.strip() if env_key else None


def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """Validate file presence, extension, and size."""
    if uploaded_file is None:
        return False, "Please upload a resume."

    filename = uploaded_file.name or ""
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported file type. Please upload PDF, DOCX, or TXT."

    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        file_size = len(uploaded_file.getvalue())

    if not file_size or file_size <= 0:
        return False, "The uploaded file is empty."

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"Maximum resume size is {MAX_FILE_SIZE_MB} MB."

    return True, ""


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise ValueError("The PDF is password protected.")
        except Exception as exc:
            raise ValueError(
                "The PDF is encrypted or password protected and cannot be read."
            ) from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue

    return "\n".join(pages)


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_txt_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def normalize_resume_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = []

    for line in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)

    return "\n".join(lines).strip()


def extract_resume_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        text = extract_pdf_text(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_docx_text(file_bytes)
    elif filename.endswith(".txt"):
        text = extract_txt_text(file_bytes)
    else:
        raise ValueError("Unsupported resume format.")

    text = normalize_resume_text(text)

    if len(text) < MIN_RESUME_CHARS:
        raise ValueError(
            "Very little readable text was extracted. The file may be "
            "image-based, empty, corrupted, or difficult to parse."
        )

    return text


def limit_text(text: str, maximum: int) -> tuple[str, bool]:
    if len(text) <= maximum:
        return text, False
    return text[:maximum], True


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "ats_score": {"type": "INTEGER"},
        "overall_summary": {"type": "STRING"},
        "job_match_summary": {"type": "STRING"},
        "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
        "weaknesses": {"type": "ARRAY", "items": {"type": "STRING"}},
        "matched_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missing_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "relevant_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "missing_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "required_qualifications": {"type": "ARRAY", "items": {"type": "STRING"}},
        "preferred_qualifications": {"type": "ARRAY", "items": {"type": "STRING"}},
        "experience_alignment": {"type": "STRING"},
        "formatting_issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        "content_issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        "section_improvements": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "section": {"type": "STRING"},
                    "issue": {"type": "STRING"},
                    "recommendation": {"type": "STRING"},
                },
                "required": ["section", "issue", "recommendation"],
            },
        },
        "actionable_recommendations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "priority_improvements": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": [
        "ats_score",
        "overall_summary",
        "job_match_summary",
        "strengths",
        "weaknesses",
        "matched_keywords",
        "missing_keywords",
        "relevant_skills",
        "missing_skills",
        "required_qualifications",
        "preferred_qualifications",
        "experience_alignment",
        "formatting_issues",
        "content_issues",
        "section_improvements",
        "actionable_recommendations",
        "priority_improvements",
    ],
}


def build_analysis_prompt(resume_text: str, job_description: str) -> str:
    if job_description.strip():
        job_context = f"""
A Job Description HAS been provided.

Perform job-specific matching using ONLY this supplied Job Description.

JOB DESCRIPTION:
----------------
{job_description}
----------------
"""
    else:
        job_context = """
A Job Description has NOT been provided.

Do NOT perform or pretend to perform job-specific keyword matching.

For matched_keywords, missing_keywords, relevant_skills, missing_skills,
required_qualifications, and preferred_qualifications, return empty arrays
where job-specific information cannot be determined.

The job_match_summary must clearly state that no Job Description was provided
and job-specific matching was not performed.
"""

    return f"""
You are an expert resume evaluator specializing in ATS systems, recruiting,
resume quality, and job-description matching.

Analyze ONLY the supplied resume and, when present, the supplied Job Description.

STRICT RULES:
1. Never invent employment history, employers, titles, skills, certifications,
   education, degrees, achievements, metrics, projects, responsibilities,
   technologies, or qualifications.
2. If information is absent, do not assume it exists.
3. Recommendations can explain what the candidate could improve, but must never
   present suggestions as facts already present in the resume.
4. When a Job Description is supplied, distinguish between keywords actually
   present in the resume and important keywords present in the Job Description
   but absent from the resume.
5. Do not recommend dishonest keyword stuffing.
6. The ATS score is an AI estimate from 0 to 100, not a real score from
   Workday, Greenhouse, Lever, Taleo, or any specific ATS.
7. Consider ATS compatibility, clarity, organization, relevant skills,
   experience relevance, keyword alignment when a Job Description exists,
   measurable achievements, action verbs, content quality, consistency,
   and readability.
8. Do not claim a visual formatting problem definitely exists if it cannot be
   determined from extracted text.
9. Keep recommendations practical, truthful, and actionable.
10. Prioritize the most important improvements.

{job_context}

RESUME:
--------
{resume_text}
--------

Return the analysis using the required JSON schema.
"""


def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
) -> dict[str, Any]:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_analysis_prompt(resume_text, job_description),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    response_text = getattr(response, "text", None)

    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned invalid structured data.") from exc

    validate_analysis_result(result)
    return result


REQUIRED_FIELDS = [
    "ats_score",
    "overall_summary",
    "job_match_summary",
    "strengths",
    "weaknesses",
    "matched_keywords",
    "missing_keywords",
    "relevant_skills",
    "missing_skills",
    "required_qualifications",
    "preferred_qualifications",
    "experience_alignment",
    "formatting_issues",
    "content_issues",
    "section_improvements",
    "actionable_recommendations",
    "priority_improvements",
]


def validate_analysis_result(result: Any) -> None:
    if not isinstance(result, dict):
        raise ValueError("Gemini returned an unexpected response structure.")

    missing = [field for field in REQUIRED_FIELDS if field not in result]
    if missing:
        raise ValueError("Gemini returned an incomplete analysis.")

    score = result["ats_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
        raise ValueError("Gemini returned an invalid ATS score.")

    scalar_fields = {
        "ats_score",
        "overall_summary",
        "job_match_summary",
        "experience_alignment",
    }

    for field in REQUIRED_FIELDS:
        if field not in scalar_fields and not isinstance(result[field], list):
            raise ValueError(f"Invalid response format for '{field}'.")


def score_label(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Needs Improvement"
    return "High Priority Improvement"


def display_bullets(items: list[Any], empty_message: str = "None identified.") -> None:
    valid_items = [str(item).strip() for item in items if str(item).strip()]
    if not valid_items:
        st.info(empty_message)
        return

    for item in valid_items:
        st.markdown(f"- {item}")


def render_score(score: int) -> None:
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-number">{score}/100</div>
            <div class="score-label">
                AI-Estimated ATS Score · {score_label(score)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_improvements(items: list[Any]) -> None:
    if not items:
        st.info("No section-specific improvements were identified.")
        return

    for item in items:
        if not isinstance(item, dict):
            continue

        section = str(item.get("section", "Section"))
        issue = str(item.get("issue", "")).strip()
        recommendation = str(item.get("recommendation", "")).strip()

        with st.expander(section):
            if issue:
                st.markdown("**Issue**")
                st.write(issue)

            if recommendation:
                st.markdown("**Recommendation**")
                st.write(recommendation)


def render_results(result: dict[str, Any], has_job_description: bool) -> None:
    st.divider()

    st.subheader("ATS Score")
    render_score(result["ats_score"])

    st.caption(
        "This is an AI-estimated score, not a guaranteed score from any "
        "specific employer ATS."
    )

    st.subheader("Overall Summary")
    st.write(result["overall_summary"])

    st.subheader("Priority Improvements")
    if result["priority_improvements"]:
        for index, item in enumerate(result["priority_improvements"], start=1):
            st.markdown(f"**{index}.** {item}")
    else:
        st.info("No priority improvements were identified.")

    st.subheader("Strengths")
    with st.expander("View strengths", expanded=True):
        display_bullets(result["strengths"])

    st.subheader("Weaknesses")
    with st.expander("View weaknesses", expanded=True):
        display_bullets(result["weaknesses"])

    st.subheader("Job Match")
    if has_job_description:
        st.write(result["job_match_summary"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Matched Keywords")
            display_bullets(
                result["matched_keywords"],
                "No matching keywords were identified.",
            )
        with col2:
            st.markdown("### Missing Keywords")
            display_bullets(
                result["missing_keywords"],
                "No important missing keywords were identified.",
            )

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Relevant Skills")
            display_bullets(result["relevant_skills"])
        with col4:
            st.markdown("### Missing Skills")
            display_bullets(
                result["missing_skills"],
                "No missing job-specific skills were identified.",
            )

        st.markdown("### Required Qualifications")
        display_bullets(result["required_qualifications"])

        st.markdown("### Preferred Qualifications")
        display_bullets(result["preferred_qualifications"])

        st.markdown("### Experience Alignment")
        st.write(result["experience_alignment"])
    else:
        st.info(
            "No Job Description was provided. Job-specific keyword matching "
            "and job-specific qualification analysis were not performed."
        )
        st.write(result["job_match_summary"])

    st.subheader("Formatting / ATS Compatibility")
    with st.expander("Formatting and parsing considerations"):
        display_bullets(
            result["formatting_issues"],
            "No specific formatting issue was identified from the extracted text.",
        )
        st.caption(
            "Text extraction cannot reliably expose every visual formatting "
            "characteristic of a PDF or DOCX."
        )

    st.subheader("Content Issues")
    with st.expander("Content quality issues"):
        display_bullets(result["content_issues"])

    st.subheader("Section-by-Section Improvements")
    render_section_improvements(result["section_improvements"])

    st.subheader("Actionable Recommendations")
    with st.expander("Recommended actions", expanded=True):
        display_bullets(result["actionable_recommendations"])


def main() -> None:
    st.markdown(
        '<div class="main-title">📄 AI Resume ATS Analyzer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Analyze your resume for ATS compatibility, resume quality, "
        "and job-description alignment using Gemini."
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "ATS score disclaimer: This is an AI-estimated score. It does not "
        "represent the exact behavior of Workday, Greenhouse, Lever, Taleo, "
        "or any other specific ATS."
    )

    api_key = get_api_key()

    if not api_key:
        st.error(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Secrets "
            "or your environment variables before analyzing a resume."
        )
        st.stop()

    st.header("1. Upload Your Resume")

    uploaded_file = st.file_uploader(
        "Choose your resume",
        type=SUPPORTED_EXTENSIONS,
        help=f"Supported: PDF, DOCX, TXT. Maximum size: {MAX_FILE_SIZE_MB} MB.",
    )

    st.caption(
        f"Supported formats: PDF, DOCX, TXT · Maximum size: {MAX_FILE_SIZE_MB} MB"
    )

    st.header("2. Job Description")

    job_description = st.text_area(
        "Paste the Job Description (optional)",
        height=240,
        max_chars=MAX_JOB_DESCRIPTION_CHARS,
        placeholder=(
            "Paste the job description here for job-specific keyword, "
            "skills, qualifications, and relevance analysis..."
        ),
    )

    st.header("3. Analyze")

    analyze_button = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
    )

    if not analyze_button:
        st.markdown(
            """
            ### How it works

            1. Upload your resume.
            2. Optionally paste a Job Description.
            3. The app extracts readable resume text.
            4. Gemini analyzes the resume.
            5. Results appear in structured sections.

            **Privacy note:** The app does not intentionally save resumes to a
            database, external storage, or persistent history. Resume text is
            sent to Gemini for the requested analysis.
            """
        )
        return

    valid, message = validate_uploaded_file(uploaded_file)
    if not valid:
        st.error(message)
        return

    try:
        with st.spinner("Extracting resume text..."):
            resume_text = extract_resume_text(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error(
            "The resume could not be processed. The file may be corrupted "
            "or use an unsupported structure."
        )
        return

    resume_text, resume_truncated = limit_text(resume_text, MAX_RESUME_CHARS)
    if resume_truncated:
        st.warning(
            "The extracted resume text exceeded the supported analysis limit. "
            "Only the first 50,000 characters were analyzed."
        )

    job_description, job_truncated = limit_text(
        job_description.strip(),
        MAX_JOB_DESCRIPTION_CHARS,
    )

    if job_truncated:
        st.warning(
            "The Job Description exceeded the supported limit and was truncated."
        )

    try:
        with st.spinner("Analyzing your resume with Gemini Flash..."):
            result = analyze_resume(
                resume_text=resume_text,
                job_description=job_description,
                api_key=api_key,
            )
    except Exception as exc:
        error_text = str(exc).lower()

        if any(
            term in error_text
            for term in [
                "api key",
                "authentication",
                "unauthenticated",
                "permission",
                "401",
                "403",
            ]
        ):
            st.error(
                "Gemini authentication failed. Please verify your "
                "GEMINI_API_KEY."
            )
        elif any(
            term in error_text
            for term in ["quota", "rate limit", "429"]
        ):
            st.error(
                "Gemini's API rate limit or quota was reached. "
                "Please wait and try again later."
            )
        elif any(
            term in error_text
            for term in ["timeout", "timed out", "deadline"]
        ):
            st.error("The Gemini request timed out. Please try again.")
        elif any(
            term in error_text
            for term in ["network", "connection"]
        ):
            st.error(
                "A network error occurred while contacting Gemini. "
                "Please try again."
            )
        else:
            st.error(
                "The AI analysis could not be completed. Please try again. "
                "Check your Gemini configuration if the problem continues."
            )
        return

    render_results(
        result=result,
        has_job_description=bool(job_description.strip()),
    )


if __name__ == "__main__":
    main()
