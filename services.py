import os
from dotenv import load_dotenv
import requests
from textblob import TextBlob, Word
from typing import Tuple, List
from fpdf import FPDF
from io import BytesIO
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Configuration
HF_API_KEY = os.getenv('HF_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 500))

# Initialize spell checker
spell = Word('en')

def check_spelling(text: str) -> Tuple[str, List[str]]:
    """Enhanced spell checking with TextBlob"""
    try:
        if not text.strip():
            return text, []
            
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return corrected, [
            str(word) for i, word in enumerate(blob.words)
            if i < len(TextBlob(corrected).words) and word != TextBlob(corrected).words[i]
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spell check error: {str(e)}")

def generate_text(prompt: str, max_length: int = None, temperature: float = None) -> str:
    """Generate text using Hugging Face API"""
    try:
        if not HF_API_KEY:
            raise ValueError("Hugging Face API key not configured")
            
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": min(max_length or MAX_TEXT_LENGTH, MAX_TEXT_LENGTH),
                    "temperature": temperature or float(os.getenv('DEFAULT_TEMPERATURE')),
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
            raise HTTPException(status_code=503, detail=f"Model loading, please wait {wait_time} seconds")
            
        response.raise_for_status()
        return response.json()[0].get('generated_text', prompt)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_pdf(data: dict) -> bytes:
    """Generate PDF resume with configured settings"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font(
            os.getenv('PDF_FONT', 'Arial'),
            size=int(os.getenv('PDF_FONT_SIZE', 12))
        )
        
        # Title
        pdf.set_font(size=int(os.getenv('PDF_TITLE_SIZE', 24)))
        pdf.cell(0, 10, data['name'], ln=1, align='C')
        
        # Content
        pdf.set_font(size=int(os.getenv('PDF_FONT_SIZE', 12)))
        pdf.multi_cell(0, 10, data['summary'])
        
        # Add sections (experiences, education, skills)
        for section in ['experiences', 'education', 'skills']:
            pdf.ln(10)
            pdf.set_font(style='B')
            pdf.cell(0, 10, section.capitalize(), ln=1)
            pdf.set_font(style='')
            for item in data.get(section, []):
                pdf.multi_cell(0, 10, f"- {item}")
                
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")