from pydantic import BaseModel
from typing import List, Optional

class TextInput(BaseModel):
    text: str
    max_length: Optional[int] = None
    temperature: Optional[float] = None

class SpellCheckInput(BaseModel):
    text: str

class SpellCheckResult(BaseModel):
    original_text: str
    corrected_text: str
    misspelled_words: List[str]

class ResumeInput(BaseModel):
    name: str
    summary: str
    experiences: List[str]
    education: List[str]
    skills: List[str]