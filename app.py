import streamlit as st
import io
import os
import json
from pypdf import PdfReader
import docx
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Ultimate Pydantic Schemas for AI Output
# ---------------------------------------------------------
class SectionImprovement(BaseModel):
    section: str = Field(description="Name of the section (e.g., 'Experience')")
    improvement: str = Field(description="Actionable improvement feedback.")

class BulletPointOptimization(BaseModel):
    original: str = Field(description="Original weak bullet point.")
    optimized: str = Field(description="Rewritten, high-impact bullet point.")

class CourseRecommendation(BaseModel):
    skill: str = Field(description="The missing skill (e.g., 'AWS', 'Python').")
    recommendation: str = Field(description="Suggested topics to search on YouTube/Coursera.")

class ResumeAnalysis(BaseModel):
    # Scores
    ats_score: int = Field(description="Overall ATS compatibility score (0-100).")
    keyword_score: int = Field(description="Keyword match sub-score (0-100).")
    formatting_score: int = Field(description="ATS parsing safety sub-score (0-100).")
    impact_score: int = Field(description="Action verbs and metrics sub-score (0-100).")
    tone_score: int = Field(description="Professionalism, grammar, and readability score (0-100).")
    
    # Text Summaries
    overall_summary: str = Field(description="Overall resume quality summary.")
    priority_improvements: list[str] = Field(description="Top 3 immediate actions.")
    strengths: list[str] = Field(description="Max 5 strengths.")
    weaknesses: list[str] = Field(description="Max 5 weaknesses.")
    
    # Deep Dives
    matched_keywords: list[str] = Field(description="Found JD keywords.")
    missing_keywords: list[str] = Field(description="Missing JD keywords.")
    formatting_issues: list[str] = Field(description="Structure/Parsing issues.")
    grammar_issues: list[str] = Field(description="Specific grammar, passive voice, or spelling errors found.")
    section_improvements: list[SectionImprovement] = Field(description="Section-specific feedback.")
    optimized_bullet_points: list[BulletPointOptimization] = Field(description="Original vs optimized bullets.")
    
    # Advanced SaaS Features
    linkedin_feedback: str = Field(description="Feedback aligning the resume with the provided LinkedIn text. Output 'No LinkedIn provided' if empty.")
    interview_questions: list[str] = Field(description="5 highly probable interview questions based on profile and gaps.")
    course_recommendations: list[CourseRecommendation] = Field(description="Learning paths for missing skills.")
    cover_letter: str = Field(description="A professional cover letter.")
    full_optimized_resume_markdown: str = Field(description="A completely rewritten, highly optimized, ATS-friendly version of the resume in Markdown format.")
    
    # New: Ultimate Career Suite Features
    localization_feedback: str = Field(description="Specific feedback based on the target country's resume standards (e.g., photo rules, length, format).")
    cold_emails: list[str] = Field(description="3 tailored cold email templates for outreach (Short, Professional, Creative).")
    portfolio_projects: list[str] = Field(description="2-3 mini-project ideas the user can build to cover missing skills in the JD.")
    salary_script: str = Field(description="A script and strategy for negotiating salary based on the role and experience level.")
    career_roadmap: list[str] = Field(description="A step-by-step 3-year career progression roadmap based on the current resume.")

# ---------------------------------------------------------
# 2. File Processing Functions
# ---------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    except:
        raise ValueError("Failed to read PDF. It might be image-based or corrupted.")

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs]).strip()
    except:
        raise ValueError("Failed to read DOCX file.")

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8').strip()
    except:
        raise ValueError("Failed to read TXT file.")

def extract_text(file_obj):
    name, data = file_obj.name.lower(), file_obj.read()
    if name.endswith('.pdf'): return extract_text_from_pdf(data)
    elif name.endswith('.docx'): return extract_text_from_docx(data)
    elif name.endswith('.txt'): return extract_text_from_txt(data)
    raise ValueError("Unsupported format.")

# ---------------------------------------------------------
# 3. Secure API Handling & Gemini Integration
# ---------------------------------------------------------
def get_api_key():
    return st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

def analyze_resume(resume_text, job_desc, linkedin_text, ats_platform, target_region, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an elite Global Tech Recruiter, ATS parsing algorithm for '{ats_platform}', and Senior Career Coach.
    Analyze the following resume deeply based on {ats_platform} rules and {target_region} corporate standards.
    """
    
    if job_desc.strip(): prompt += f"\nTarget Job Description:\n{job_desc}\n"
    if linkedin_text.strip(): prompt += f"\nCandidate's LinkedIn Text:\n{linkedin_text}\n"
    
    prompt += f"\nCandidate Resume Text:\n{resume_text}\n"
    prompt += f"""
    CRITICAL INSTRUCTIONS:
    1. Score strictly out of 100 based on standard algorithms.
    2. Identify specific grammar and readability issues.
    3. Generate 5 behavioral/technical interview questions based on their profile.
    4. Provide course/learning suggestions for missing skills.
    5. Rewrite the ENTIRE resume beautifully in Markdown format using impact metrics.
    6. Provide localization feedback specific to {target_region} (e.g., rules about photos, dates, length).
    7. Generate 3 networking cold emails (Short, Professional, Creative).
    8. Suggest 2-3 portfolio mini-projects to fill skill gaps.
    9. Write a tactical salary negotiation script.
    10. Map out a 3-year career progression roadmap.
    11. Return ONLY valid JSON adhering to the schema. Do not invent experience.
    """
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeAnalysis,
            temperature=0.2, 
        )
    )
    return json.loads(response.text)

# ---------------------------------------------------------
# 4. Helper UI Functions
# ---------------------------------------------------------
def render_progress_bar(label, value):
    color = "#28a745" if value >= 80 else "#ffc107" if value >= 50 else "#dc3545"
    st.markdown(f"**{label}: {value}/100**")
    st.markdown(f"""
        <div style="width:100%; background:#e0e0e0; border-radius:5px; margin-bottom:15px;">
            <div style="width:{value}%; background:{color}; height:12px; border-radius:5px;"></div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Streamlit UI Components
# ---------------------------------------------------------
st.set_page_config(page_title="AI-Powered Resume Intelligence", page_icon="📄", layout="wide")
st.title("📄 AI-Powered Resume Intelligence")
st.markdown("Diagnose, rewrite, prep for interviews, network, and plan your career trajectory in one powerful platform.")

use_demo = st.checkbox("🧪 Try Sample Data (Demo Mode)")

if use_demo:
    resume_input_text = "John Doe\nSoftware Developer at TechCorp\n- Wrote python code.\n- Fixed bugs."
    job_desc = "Seeking a Backend Dev with AWS, Python, and scalable architecture experience."
    linkedin_input = "I am a programmer looking for new jobs."
    uploaded_file = None
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Upload Resume")
        uploaded_file = st.file_uploader("Upload file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
        ats_sys = st.selectbox("🎯 Target ATS Simulator", ["Generic", "Workday", "Taleo", "Greenhouse", "Lever"])
    with col2:
        st.markdown("### 2. Add Context (Optional)")
        target_region = st.selectbox("🌍 Target Region", ["US/Canada", "Europe/UK", "Gulf/Middle East", "Asia/Australia", "Remote/Global"])
        job_desc = st.text_area("Target Job Description", height=100)
        linkedin_input = st.text_area("LinkedIn About/Profile Text", height=100)

if st.button("🔥 Run Ultimate Analysis", type="primary", use_container_width=True):
    api_key = get_api_key()
    if not api_key:
        st.error("🚨 GEMINI_API_KEY is missing in Secrets.")
        st.stop()
        
    if not use_demo and not uploaded_file:
        st.error("Please upload a resume.")
        st.stop()

    try:
        with st.spinner("Executing multi-agent ATS scanning..."):
            text = resume_input_text if use_demo else extract_text(uploaded_file)
            if len(text) > 20000: text = text[:20000]

        with st.spinner("Running deep AI diagnostics (Outreach, Projects, Career Planning)... Approx 15-20 sec."):
            sys = "Generic" if use_demo else ats_sys
            region = "US/Canada" if use_demo else target_region
            result = analyze_resume(text, job_desc, linkedin_input, sys, region, api_key)
            
        # --- DASHBOARD RENDERING ---
        st.success("✅ Analysis Complete!")
        st.markdown("---")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            render_progress_bar("Overall ATS Score", result['ats_score'])
            render_progress_bar("Keyword Match", result['keyword_score'])
            st.markdown("#### 🚨 Priority Actions")
            for p in result['priority_improvements'][:3]: st.markdown(f"- {p}")
        with c2:
            render_progress_bar("Impact & Metrics", result['impact_score'])
            render_progress_bar("Tone & Grammar", result['tone_score'])
            render_progress_bar("Formatting Safety", result['formatting_score'])

        st.markdown("---")
        
        # We now have 8 comprehensive tabs
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "📊 Core Diagnostics", 
            "📄 Auto-Built Resume", 
            "🌍 Localization",
            "🤝 Outreach & Salary",
            "🚀 Projects & Career",
            "🗣️ Interview Prep", 
            "📚 Skill Gaps", 
            "✉️ LinkedIn & Cover Letter"
        ])
        
        with t1:
            st.markdown(f"**Executive Summary:** {result['overall_summary']}")
            if result['grammar_issues']:
                st.warning("**Grammar & Readability Issues Found:**")
                for g in result['grammar_issues']: st.write(f"- {g}")
            cc1, cc2 = st.columns(2)
            with cc1:
                st.success("Matched Keywords")
                for m in result['matched_keywords']: st.write(f"- {m}")
            with cc2:
                st.error("Missing Keywords")
                for m in result['missing_keywords']: st.write(f"- {m}")

        with t2:
            st.markdown("#### ✨ Before vs After Bullet Points")
            for item in result['optimized_bullet_points']:
                st.error(f"**Weak:** {item['original']}")
                st.success(f"**Pro:** {item['optimized']}")
            st.markdown("---")
            st.markdown("#### 📄 Your New Auto-Generated Resume")
            st.info("Copy this Markdown format directly into Notion or MS Word (as plain text).")
            st.markdown(result['full_optimized_resume_markdown'])
            st.download_button("📥 Download Resume (.md)", result['full_optimized_resume_markdown'], "optimized_resume.md")

        with t3:
            st.markdown(f"#### 🌍 Regional Advice: {region if not use_demo else 'US/Canada'}")
            st.info(result['localization_feedback'])

        with t4:
            st.markdown("#### ✉️ Networking & Cold Emails")
            st.write("Use these templates to reach out to recruiters or hiring managers on LinkedIn or via email.")
            for i, email in enumerate(result['cold_emails'], 1):
                with st.expander(f"Email Template {i}"):
                    st.code(email, language="text")
            
            st.markdown("#### 💰 Salary Negotiation Script")
            st.success(result['salary_script'])

        with t5:
            st.markdown("#### 🛠️ Portfolio Mini-Projects")
            st.write("Build these small projects to cover missing skills from the job description:")
            for p in result['portfolio_projects']:
                st.markdown(f"- {p}")
            
            st.markdown("#### 🗺️ 3-Year Career Roadmap")
            for step in result['career_roadmap']:
                st.markdown(f"👉 {step}")

        with t6:
            st.markdown("#### 🗣️ AI Mock Interview Questions")
            for i, q in enumerate(result['interview_questions'], 1):
                st.markdown(f"**{i}.** {q}")

        with t7:
            st.markdown("#### 📚 Learning Paths for Missing Skills")
            for cr in result['course_recommendations']:
                st.markdown(f"- **{cr['skill']}**: {cr['recommendation']}")

        with t8:
            st.markdown("#### 🌐 LinkedIn Profile Feedback")
            st.write(result['linkedin_feedback'])
            st.markdown("#### ✉️ Tailored Cover Letter")
            st.text_area("Cover Letter", result['cover_letter'], height=200)
            st.download_button("📥 Download Cover Letter", result['cover_letter'], "cover_letter.txt")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")