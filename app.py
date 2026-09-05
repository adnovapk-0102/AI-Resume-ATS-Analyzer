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
# 1. Pydantic Schemas for Structured AI Output
# ---------------------------------------------------------
class SectionImprovement(BaseModel):
    section: str = Field(description="Name of the section, e.g., 'Work Experience' or 'Education'")
    improvement: str = Field(description="Suggested actionable improvement for this section based strictly on provided content")

class ResumeAnalysis(BaseModel):
    ats_score: int = Field(description="Overall ATS compatibility score from 0 to 100.")
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
    You are an expert Applicant Tracking System (ATS) software analyzer and senior tech recruiter.
    Analyze the following extracted resume text.
    """
    
    if job_description.strip():
        prompt += f"""
        A job description has been provided. Evaluate how well the resume matches this specific role.
        Identify matched keywords (skills, tools, qualifications found in BOTH) and missing keywords (found in JD but MISSING from resume).
        
        Job Description:
        {job_description}
        """
    else:
        prompt += """
        No job description was provided. Evaluate the resume against general industry best practices, 
        standard ATS parsing rules, readability, and overall resume quality. Do not fabricate job match data.
        """
        
    prompt += f"""
    Resume Text:
    {resume_text}
    
    CRITICAL INSTRUCTIONS:
    1. Base all feedback strictly on the provided resume text.
    2. DO NOT invent, hallucinate, or assume employment history, skills, certifications, education, or achievements.
    3. Clearly distinguish between what is present and what is missing.
    4. Provide actionable, practical feedback for improvement.
    5. Evaluate formatting based on how the text parsed (e.g., lack of clear section headers may indicate poor ATS compatibility).
    6. Ensure the ATS score reflects an objective 0-100 estimate based on relevance and parseability.
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
st.set_page_config(page_title="AI Resume ATS Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI-Powered Resume ATS Analyzer")
st.markdown("""
Welcome! This application extracts text from your resume and uses Google Gemini Flash to analyze it like an Applicant Tracking System (ATS). 
Upload your resume, optionally paste a target Job Description, and receive actionable, structured feedback to improve your chances.
""")

st.warning("**Disclaimer:** The ATS score is an AI-generated estimate based on standard industry parsing rules. It is not a guarantee of how specific enterprise platforms (like Workday, Greenhouse, or Lever) will rank your resume.", icon="⚠️")

# UI Inputs
st.markdown("### 1. Upload Your Resume")
uploaded_file = st.file_uploader("Upload a file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

st.markdown("### 2. Job Description (Optional)")
job_desc = st.text_area("Paste the job description here for tailored match analysis (leave blank for a general analysis).", height=150)

# Process Trigger
if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
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
        with st.spinner("Extracting text from document..."):
            resume_text = extract_text(uploaded_file)
            
            if len(resume_text) < 100:
                st.error("Could not extract sufficient text. If you uploaded an image-based PDF, please use a text-based format.")
                st.stop()
            
            if len(resume_text) > 20000:
                st.warning("Resume is unusually long. It has been truncated to 20,000 characters to optimize AI analysis.")
                resume_text = resume_text[:20000]

        with st.spinner("Analyzing resume with Gemini Flash... This may take a few seconds."):
            result = analyze_resume(resume_text, job_desc, api_key)
            
        # ---------------------------------------------------------
        # 5. Render Structured Results
        # ---------------------------------------------------------
        st.success("✅ Analysis Complete!")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Estimated ATS Score", f"{result['ats_score']} / 100")
            st.markdown("#### 🚨 Priority Improvements")
            for item in result['priority_improvements']:
                st.markdown(f"- **{item}**")
        with col2:
            st.markdown("#### 📊 Overall Summary")
            st.write(result['overall_summary'])

        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Keywords & Matching", "Strengths & Weaknesses", "Formatting & Content", "Section Feedback"])
        
        with tab1:
            if job_desc.strip():
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ Matched Keywords")
                    if result['matched_keywords']:
                        for mk in result['matched_keywords']:
                            st.markdown(f"- {mk}")
                    else:
                        st.write("No strong matches found.")
                with c2:
                    st.error("❌ Missing Keywords")
                    if result['missing_keywords']:
                        for msk in result['missing_keywords']:
                            st.markdown(f"- {msk}")
                    else:
                        st.write("No major keywords missing!")
            else:
                st.info("ℹ️ No Job Description was provided. Keyword matching was skipped. The analysis reflects general ATS optimization.")
                
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 💪 Resume Strengths")
                for s in result['strengths']: st.markdown(f"- {s}")
            with c2:
                st.markdown("#### 📉 Areas for Improvement")
                for w in result['weaknesses']: st.markdown(f"- {w}")
                
        with tab3:
            st.markdown("#### 📑 Formatting Issues (ATS Parsing)")
            if result['formatting_issues']:
                for f in result['formatting_issues']: st.markdown(f"- {f}")
            else:
                st.write("No major formatting issues detected.")
                
            st.markdown("#### 📝 Content Issues")
            if result['content_issues']:
                for c in result['content_issues']: st.markdown(f"- {c}")
            else:
                st.write("No major content issues detected.")
                
            st.markdown("#### 🛠️ Actionable Recommendations")
            for r in result['actionable_recommendations']: st.markdown(f"- {r}")

        with tab4:
            st.markdown("#### 🔍 Section-by-Section Recommendations")
            if result['section_improvements']:
                for section in result['section_improvements']:
                    with st.expander(f"Section: {section['section']}"):
                        st.write(section['improvement'])
            else:
                st.write("No specific section improvements needed.")

    except ValueError as ve:
        st.error(f"File Processing Error: {str(ve)}")
    except Exception as e:
        st.error(f"An error occurred during AI analysis. Error details: {str(e)}")
