import streamlit as st
from datetime import datetime
from fpdf import FPDF
import base64
import requests
import json
from textblob import TextBlob
from textblob import Word
from dotenv import load_dotenv
import os
from typing import Tuple, List

# Load environment variables
load_dotenv()

# Configuration
HF_API_KEY = os.getenv('HF_API_KEY')
DEFAULT_TEMPLATE = os.getenv('DEFAULT_TEMPLATE', 'Professional')
PDF_FONT = os.getenv('PDF_FONT', 'Arial')
PDF_FONT_SIZE = int(os.getenv('PDF_FONT_SIZE', 12))
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 500))
MODEL_NAME = os.getenv('MODEL_NAME', 'google/flan-t5-large')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))

# Initialize spell checker
spell = Word('en')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Resume Builder", 
    layout="centered", 
    page_icon="🤖"
)
st.title("🤖 AI-Powered Resume Builder")

# Initialize session state
if 'education_entries' not in st.session_state:
    st.session_state.education_entries = [{
        "degree": "", 
        "school": "", 
        "year": "", 
        "achievements": "", 
        "gpa": ""
    }]
    
if 'experience_entries' not in st.session_state:
    st.session_state.experience_entries = [{
        "role": "", 
        "company": "", 
        "years": "", 
        "months": 0, 
        "description": "", 
        "achievements": "", 
        "location": ""
    }]

if 'ai_suggestions' not in st.session_state:
    st.session_state.ai_suggestions = {}
    
if 'ai_enhanced' not in st.session_state:
    st.session_state.ai_enhanced = False

# Core Functions
def check_spelling(text: str) -> Tuple[str, List[str]]:
    """Check spelling and return corrected text with misspelled words"""
    if not text.strip():
        return text, []
    
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return corrected, [
            str(word) for i, word in enumerate(blob.words)
            if i < len(TextBlob(corrected).words) and word != TextBlob(corrected).words[i]
        ]
    except Exception as e:
        st.error(f"Spell check error: {str(e)}")
        return text, []

def generate_text(prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
    """Generate text using Hugging Face API"""
    if not HF_API_KEY:
        st.error("Hugging Face API key not configured")
        return ""
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_length, MAX_TEXT_LENGTH),
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": 0.9,
                    "return_full_text": False
                },
                "options": {"wait_for_model": True}
            },
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 503:
            wait_time = int(response.headers.get("estimated_time", 30))
            st.warning(f"Model is loading. Please wait {wait_time} seconds.")
            return ""
            
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get('generated_text', prompt)
        return prompt
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Text generation error: {str(e)}")
        return ""

def create_pdf(data: dict) -> bytes:
    """Generate PDF resume"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font(PDF_FONT, size=PDF_FONT_SIZE)
        
        # Header
        pdf.set_font(size=24)
        pdf.cell(0, 10, data['name'], ln=1, align='C')
        pdf.set_font(size=12)
        pdf.cell(0, 10, data.get('contact', ''), ln=1, align='C')
        pdf.ln(10)
        
        # Summary
        pdf.set_font(style='B')
        pdf.cell(0, 10, 'Professional Summary', ln=1)
        pdf.set_font(style='')
        pdf.multi_cell(0, 10, data['summary'])
        pdf.ln(10)
        
        # Sections
        for section in ['education', 'experience']:
            pdf.set_font(style='B')
            pdf.cell(0, 10, section.capitalize(), ln=1)
            pdf.set_font(style='')
            
            for entry in data.get(section, []):
                if section == 'education':
                    text = f"{entry['degree']} | {entry['school']} | {entry['year']}"
                    if entry.get('gpa'):
                        text += f" | GPA: {entry['gpa']}"
                else:
                    text = f"{entry['role']} | {entry['company']} | {entry['years']} years"
                    if entry.get('location'):
                        text += f" | {entry['location']}"
                
                pdf.multi_cell(0, 10, text)
                if entry.get('description'):
                    pdf.multi_cell(0, 10, entry['description'])
                if entry.get('achievements'):
                    pdf.multi_cell(0, 10, f"Achievements: {entry['achievements']}")
                pdf.ln(5)
                
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

# Sidebar with AI Features
with st.sidebar:
    st.header("🔮 AI Features")
    
    ai_feature = st.selectbox(
        "Select AI Assistance",
        ["None", "Spell Check", "Text Generator"],
        help="Choose how you want AI to help with your resume"
    )
    
    if ai_feature == "Spell Check":
        st.info("Check spelling in your resume content")
        if st.button("Run Spell Check"):
            with st.spinner("🔍 Checking spelling..."):
                spelling_errors = {}
                
                # Check summary
                if 'summary' in st.session_state and st.session_state.summary:
                    corrected, errors = check_spelling(st.session_state.summary)
                    if errors:
                        spelling_errors['summary'] = (corrected, errors)
                
                # Check skills
                if 'technical_skills' in st.session_state and st.session_state.technical_skills:
                    corrected, errors = check_spelling(st.session_state.technical_skills)
                    if errors:
                        spelling_errors['skills'] = (corrected, errors)
                
                # Check experience descriptions
                for i, exp in enumerate(st.session_state.experience_entries):
                    if exp["description"]:
                        corrected, errors = check_spelling(exp["description"])
                        if errors:
                            spelling_errors[f'exp_desc_{i}'] = (corrected, errors)
                
                if spelling_errors:
                    st.session_state.spelling_errors = spelling_errors
                    st.warning("Spelling errors found! See suggestions below.")
                else:
                    st.success("No spelling errors found!")
    
    elif ai_feature == "Text Generator":
        st.info("Generate any text for your resume")
        with st.form("text_generator_form"):
            text_prompt = st.text_area(
                "Enter your text generation prompt:", 
                placeholder="e.g., Write a professional summary for a data scientist with 5 years experience...",
                height=150
            )
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.slider("Max Length", 50, 500, 200)
            with col2:
                temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 
                                      help="Lower for more factual, higher for more creative")
            
            submitted = st.form_submit_button("Generate Text")
            
            if submitted and text_prompt:
                with st.spinner("🧠 Generating text..."):
                    generated_text = generate_text(text_prompt, max_length, temperature)
                    if generated_text:
                        st.session_state.generated_text = generated_text
    
    # Personal Details Section
    st.header("Personal Details")
    st.session_state.name = st.text_input("Full Name*", placeholder="John Doe")
    st.session_state.email = st.text_input("Email*", placeholder="john@example.com")
    st.session_state.phone = st.text_input("Phone", placeholder="+1 (123) 456-7890")
    st.session_state.linkedin = st.text_input("LinkedIn URL", placeholder="https://linkedin.com/in/username")
    st.session_state.github = st.text_input("GitHub URL", placeholder="https://github.com/username")
    st.session_state.portfolio = st.text_input("Portfolio URL", placeholder="https://yourportfolio.com")
    st.session_state.address = st.text_input("Address", placeholder="City, Country")
    
    st.header("Career Summary")
    st.session_state.summary = st.text_area(
        "Professional Summary (3-5 sentences)*", 
        height=150
    )
    
    st.header("Skills")
    st.session_state.technical_skills = st.text_area(
        "Technical Skills", 
        placeholder="Python, Machine Learning, SQL"
    )
    st.session_state.soft_skills = st.text_area(
        "Soft Skills", 
        placeholder="Leadership, Communication"
    )
    
    st.header("Template Selection")
    st.session_state.template = st.selectbox(
        "Choose a template", 
        ["Professional Executive", "Modern Tech", "Academic", "Creative Designer"],
        help="Select a template that matches your industry"
    )

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Education Section
    st.subheader("Education")
    for i, entry in enumerate(st.session_state.education_entries):
        with st.expander(f"Education Entry #{i+1}", expanded=True):
            st.session_state.education_entries[i]["degree"] = st.text_input(
                f"Degree/Certification {i+1}", 
                value=entry["degree"],
                placeholder="e.g., B.S. in Computer Science",
                key=f"degree_{i}"
            )
            st.session_state.education_entries[i]["school"] = st.text_input(
                f"Institution {i+1}", 
                value=entry["school"],
                placeholder="e.g., University of California",
                key=f"school_{i}"
            )
            cols = st.columns(2)
            with cols[0]:
                st.session_state.education_entries[i]["year"] = st.text_input(
                    f"Year {i+1}", 
                    value=entry["year"],
                    placeholder="e.g., 2015-2019",
                    key=f"year_{i}"
                )
            with cols[1]:
                st.session_state.education_entries[i]["gpa"] = st.text_input(
                    f"GPA {i+1}", 
                    value=entry.get("gpa", ""),
                    placeholder="e.g., 3.8/4.0",
                    key=f"gpa_{i}"
                )
            st.session_state.education_entries[i]["achievements"] = st.text_area(
                f"Notable Achievements {i+1}", 
                value=entry["achievements"],
                placeholder="Honors, awards, or special projects",
                height=100,
                key=f"edu_ach_{i}"
            )

    if st.button("➕ Add Another Education Entry"):
        st.session_state.education_entries.append({
            "degree": "", 
            "school": "", 
            "year": "", 
            "achievements": "", 
            "gpa": ""
        })
        
    if len(st.session_state.education_entries) > 1 and st.button("➖ Remove Last Education Entry"):
        st.session_state.education_entries.pop()

with col2:
    # Experience Section
    st.subheader("Work Experience")
    for i, entry in enumerate(st.session_state.experience_entries):
        with st.expander(f"Experience Entry #{i+1}", expanded=True):
            st.session_state.experience_entries[i]["role"] = st.text_input(
                f"Job Title {i+1}", 
                value=entry["role"],
                placeholder="e.g., Data Scientist",
                key=f"role_{i}"
            )
            st.session_state.experience_entries[i]["company"] = st.text_input(
                f"Company {i+1}", 
                value=entry["company"],
                placeholder="e.g., Tech Corp Inc.",
                key=f"company_{i}"
            )
            cols = st.columns(2)
            with cols[0]:
                st.session_state.experience_entries[i]["years"] = st.text_input(
                    f"Duration (Years)", 
                    value=entry["years"],
                    placeholder="e.g., 2",
                    key=f"years_{i}"
                )
            with cols[1]:
                st.session_state.experience_entries[i]["months"] = st.number_input(
                    f"Duration (Months)", 
                    min_value=0,
                    max_value=11,
                    value=entry.get("months", 0),
                    key=f"months_{i}"
                )
            st.session_state.experience_entries[i]["location"] = st.text_input(
                f"Location {i+1}", 
                value=entry.get("location", ""),
                placeholder="e.g., Remote",
                key=f"location_{i}"
            )
            st.session_state.experience_entries[i]["description"] = st.text_area(
                f"Responsibilities {i+1}", 
                value=entry["description"],
                placeholder="Describe your role and responsibilities",
                height=100,
                key=f"desc_{i}"
            )
            st.session_state.experience_entries[i]["achievements"] = st.text_area(
                f"Key Achievements {i+1}", 
                value=entry["achievements"],
                placeholder="Quantifiable achievements with metrics if possible",
                height=100,
                key=f"exp_ach_{i}"
            )

    if st.button("➕ Add Another Experience Entry"):
        st.session_state.experience_entries.append({
            "role": "", 
            "company": "", 
            "years": "", 
            "months": 0, 
            "description": "", 
            "achievements": "", 
            "location": ""
        })
        
    if len(st.session_state.experience_entries) > 1 and st.button("➖ Remove Last Experience Entry"):
        st.session_state.experience_entries.pop()

# Additional Sections
st.subheader("Additional Information")
st.session_state.projects = st.text_area(
    "Projects", 
    height=150, 
    placeholder="Describe any relevant projects with technologies used and outcomes"
)
st.session_state.certifications = st.text_area(
    "Certifications", 
    height=100, 
    placeholder="List any professional certifications with dates"
)
st.session_state.languages = st.text_input(
    "Languages", 
    placeholder="e.g., English (Fluent), Spanish (Intermediate)"
)
st.session_state.publications = st.text_area(
    "Publications/Presentations", 
    height=100,
    placeholder="List any relevant publications or conference presentations"
)
st.session_state.volunteer = st.text_area(
    "Volunteer Experience", 
    height=100,
    placeholder="List any relevant volunteer work"
)

# Text Generator Panel
if st.session_state.get('generated_text'):
    with st.expander("✍️ Generated Text", expanded=True):
        st.text_area(
            "Generated Text", 
            value=st.session_state.generated_text, 
            height=200,
            key="generated_text_output"
        )

# Spelling Errors Panel
if st.session_state.get('spelling_errors'):
    with st.expander("✏️ Spelling Corrections", expanded=True):
        for field, (corrected, errors) in st.session_state.spelling_errors.items():
            st.subheader(f"Spelling errors in {field.replace('_', ' ')}")
            st.write(f"Misspelled words: {', '.join(errors)}")
            st.text_area(
                "Corrected version", 
                value=corrected, 
                height=100, 
                key=f"corrected_{field}"
            )
            if st.button(f"Apply corrections to {field}", key=f"correct_{field}"):
                if field == "summary":
                    st.session_state.summary = corrected
                elif field == "skills":
                    st.session_state.technical_skills = corrected
                elif field.startswith("exp_desc"):
                    idx = int(field.split("_")[-1])
                    st.session_state.experience_entries[idx]["description"] = corrected
                st.success("Corrections applied!")
                st.rerun()

# Generate Resume Button
if st.button("Generate Professional Resume", type="primary"):
    if not st.session_state.name or not st.session_state.email or not st.session_state.summary:
        st.error("Please fill in at least the required fields (marked with *)")
    else:
        # Prepare data for PDF
        resume_data = {
            "name": st.session_state.name,
            "contact": f"{st.session_state.email} | {st.session_state.phone} | {st.session_state.address}",
            "summary": st.session_state.summary,
            "education": st.session_state.education_entries,
            "experience": st.session_state.experience_entries,
            "skills": {
                "technical": st.session_state.technical_skills,
                "soft": st.session_state.soft_skills
            },
            "additional": {
                "projects": st.session_state.projects,
                "certifications": st.session_state.certifications,
                "languages": st.session_state.languages,
                "publications": st.session_state.publications,
                "volunteer": st.session_state.volunteer
            }
        }
        
        # Create PDF
        with st.spinner("🔄 Generating your professional resume..."):
            pdf_bytes = create_pdf(resume_data)
            
            if pdf_bytes:
                st.download_button(
                    label="📄 Download Resume as PDF",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.name.replace(' ', '_')}_Resume.pdf",
                    mime="application/pdf"
                )
                st.success("Resume generated successfully!")
            else:
                st.error("Failed to generate PDF. Please try again.")