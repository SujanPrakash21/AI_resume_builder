from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from models import TextInput, SpellCheckInput, SpellCheckResult, ResumeInput
from services import check_spelling, generate_text, create_pdf
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": os.getenv('MODEL_NAME'),
        "max_length": int(os.getenv('MAX_TEXT_LENGTH', 500))
    }

@app.post("/spell-check")
async def spell_check(input: SpellCheckInput):
    corrected_text, misspelled = check_spelling(input.text)
    return SpellCheckResult(
        original_text=input.text,
        corrected_text=corrected_text,
        misspelled_words=misspelled
    )

@app.post("/generate-text")
async def text_generation(input: TextInput):
    generated = generate_text(
        input.text,
        input.max_length,
        input.temperature
    )
    return {"generated_text": generated}

@app.post("/generate-resume", response_class=StreamingResponse)
async def resume_generation(input: ResumeInput):
    pdf_bytes = create_pdf({
        "name": input.name,
        "summary": input.summary,
        "experiences": input.experiences,
        "education": input.education,
        "skills": input.skills
    })
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={input.name.replace(' ', '_')}_Resume.pdf"
        }
    )