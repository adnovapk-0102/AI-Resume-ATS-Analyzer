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
# 1. Advanced Pydantic Schemas for Structured AI Output
# ---------------------------------------------------------
class SectionImprovement(BaseModel):
    section: str = Field(description="Name of the section, e.g., 'Work Experience' or 'Education'")
    improvement: str = Field(description="Suggested actionable improvement for this section based strictly on provided content")

class ResumeAnalysis(BaseModel):
    ats_score: int = Field(description="Overall ATS compatibility score from 0 to 100.")
    keyword_score: int = Field(description="Sub-score for keyword relevance from 0 to 100.")
    formatting_score: int = Field(description="Sub-score for ATS parsing safety and structure from 0 to 100.")
    impact_score: int = Field(description="Sub-score for action verbs and measurable achievements from 0 to 100.")
    overall_summary: str = Field(description="Summary of the resume's overall quality, readability, and relevance.")
    strengths: list[str] = Field(description="List of strengths found in the resume. Max 5.")
    weaknesses: list[str] = Field(description="List of weaknesses or areas for improvement. Max 5.")
    matched_keywords: list[str] = Field(description="Keywords from the job description matched in the resume. Empty if no JD provided.")
    missing_keywords: list[str] = Field(description="Important keywords from the job description missing in the resume. Empty if no JD provided.")
    formatting_issues: list[str] = Field(description="Potential ATS parsing issues related to structure and formatting.")
    content_issues: list[str] = Field(description="Issues related to the actual content, phrasing, or missing standard sections.")
    section_improvements: list[SectionImprovement] = Field(description="Specific actionable feedback broken down by resume section.")
    actionable_recommendations: list[str] = Field(description="Concrete steps the user can take to improve their resume overall.")
    priority_improvements: list[str] = Field(description="The top 1-3 most critical improvements to make immediately.")
    optimized_bullet_points: list[str] = Field(description="Suggested high-impact rewritten bullet points for the resume.")
    cover_letter: str = Field(description="A professional, tailored cover letter based on the resume and job description.")

# ---------------------------------------------------------
# 2. File Processing Functions
# ---------------------------------------------------------
def extract_text_from_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to read PDF file: {str(e)}")

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to read DOCX file: {str(e)}")

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8').strip()
    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {str(e)}")

def extract_text(file_obj):
    file_name = file_obj.name.lower()
    file_bytes = file_obj.read()
    
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif file_name.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError("Unsupported file type.")

# ---------------------------------------------------------
# 3. Secure API Handling & Gemini Integration
# ---------------------------------------------------------
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY")

def analyze_resume(resume_text: str, job_description: str, api_key: str):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Enterprise Applicant Tracking System (ATS) software analyzer and senior career coach.
    Analyze the following extracted resume text deeply.
    """
    
    if job_description.strip():
        prompt += f"""
        A target Job Description has been provided. Evaluate keyword matching, skill gaps, and role alignment thoroughly.
        
        Job Description:
        {job_description}
        """
    else:
        prompt += """
        No job description was provided. Evaluate the resume against global industry best practices, 
        standard parsing rules, and impact phrasing. Do not fabricate job match metrics.
        """
        
    prompt += f"""
    Resume Text:
    {resume_text}
    
    CRITICAL INSTRUCTIONS:
    1. Base all feedback strictly on the provided resume text.
    2. Provide sub-scores for keywords, formatting, and impact/achievements (0-100).
    3. Generate optimized high-impact bullet points to improve achievements.
    4. Generate a professional tailored cover letter leveraging the candidate's actual background.
    5. Ensure strict adherence to the requested JSON structure.
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
# 4. Streamlit UI Components
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced AI Resume ATS Analyzer", page_icon="⚡", layout="wide")

st.title("⚡ Advanced Enterprise-Grade AI Resume ATS Analyzer")
st.markdown("""
Welcome to the next-generation career optimization suite. Upload your resume, provide a target job description, 
and receive deep diagnostic sub-scores, automated bullet point rewrites, and a tailored professional cover letter.
""")

st.warning("**Disclaimer:** Scores and suggestions are AI-generated estimates designed to maximize resume optimization and keyword alignment.", icon="⚠️")

# UI Inputs
st.markdown("### 1. Upload Your Resume")
uploaded_file = st.file_uploader("Upload a file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

st.markdown("### 2. Job Description (Optional)")
job_desc = st.text_area("Paste the target job description for advanced alignment, keyword extraction, and cover letter generation.", height=150)

# Process Trigger
if st.button("🚀 Run Advanced AI Analysis", type="primary", use_container_width=True):
    api_key = get_api_key()
    if not api_key:
        st.error("🚨 GEMINI_API_KEY is missing! Please configure it in Streamlit Secrets.")
        st.stop()
        
    if not uploaded_file:
        st.error("Please upload a resume file first.")
        st.stop()

    MAX_FILE_SIZE = 5 * 1024 * 1024
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File size exceeds the 5MB limit. Please upload a smaller file.")
        st.stop()
        
    try:
        with st.spinner("Extracting text and running neural parser..."):
            resume_text = extract_text(uploaded_file)
            
            if len(resume_text) < 100:
                st.error("Could not extract sufficient text. Please ensure it's a readable text document.")
                st.stop()
            
            if len(resume_text) > 20000:
                resume_text = resume_text[:20000]

        with st.spinner("Analyzing parameters with Gemini Flash..."):
            result = analyze_resume(resume_text, job_desc, api_key)
            
        # ---------------------------------------------------------
        # 5. Render Advanced Results Dashboard
        # ---------------------------------------------------------
        st.success("✅ Advanced Analysis Complete!")
        st.markdown("---")
        
        # Metric Cards & Sub-Scores
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall ATS Score", f"{result['ats_score']} / 100")
        with col2:
            st.metric("Keyword Match", f"{result['keyword_score']} / 100")
        with col3:
            st.metric("Formatting Safety", f"{result['formatting_score']} / 100")
        with col4:
            st.metric("Impact & Metrics", f"{result['impact_score']} / 100")

        st.markdown("---")
        
        # Summary & Priority Focus
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            st.markdown("#### 📊 Executive Summary")
            st.write(result['overall_summary'])
        with sc2:
            st.markdown("#### 🚨 Priority Actions")
            for item in result['priority_improvements'][:3]:
                st.markdown(f"- **{item}**")

        st.markdown("---")
        
        # Multi-Tab Advanced Modules
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Keywords & Gaps", 
            "Strengths & Weaknesses", 
            "Formatting & Content", 
            "✨ AI Bullet Optimizer", 
            "✉️ Tailored Cover Letter"
        ])
        
        with tab1:
            if job_desc.strip():
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ Matched Keywords")
                    for mk in result['matched_keywords']: st.markdown(f"- {mk}")
                with c2:
                    st.error("❌ Missing Keywords")
                    for msk in result['missing_keywords']: st.markdown(f"- {msk}")
            else:
                st.info("ℹ️ Provide a Job Description above to unlock deep keyword matching analytics.")
                
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 💪 Resume Strengths")
                for s in result['strengths']: st.markdown(f"- {s}")
            with c2:
                st.markdown("#### 📉 Areas for Growth")
                for w in result['weaknesses']: st.markdown(f"- {w}")
                
        with tab3:
            st.markdown("#### 📑 Structural & Formatting Diagnostics")
            for f in result['formatting_issues']: st.markdown(f"- {f}")
            st.markdown("#### 📝 Content Refinement")
            for c in result['content_issues']: st.markdown(f"- {c}")
            
            st.markdown("#### 🔍 Section Deep Dive")
            for sec in result['section_improvements']:
                with st.expander(f"Section: {sec['section']}"):
                    st.write(sec['improvement'])

        with tab4:
            st.markdown("#### ✨ AI-Powered Optimized Bullet Points")
            st.markdown("Use these upgraded, impact-driven phrasing examples to replace weak descriptions in your experience section:")
            for bp in result['optimized_bullet_points']:
                st.info(bp)

        with tab5:
            st.markdown("#### ✉️ Generated Cover Letter")
            st.markdown("A customized, professional cover letter tailored to your profile and the target role:")
            st.text_area("Cover Letter Output", value=result['cover_letter'], height=300)

    except ValueError as ve:
        st.error(f"File Processing Error: {str(ve)}")
    except Exception as e:
        st.error(f"An error occurred during execution: {str(e)}")