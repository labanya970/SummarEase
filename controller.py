from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from summarize_model import generate_quality_summary
from highlight import extract_keywords, extract_sentences_with_keywords
from grammar_check import grammar_correct

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (you can restrict to specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TextRequest(BaseModel):
    text: str
    top_n: int = 10

# Route: Generate enhanced summary
@app.post("/summarize")
def summarize_endpoint(request: TextRequest):
    summary = generate_quality_summary(request.text)
    return {"summary": summary}

# Route: Extract top keywords and relevant sentences
@app.post("/keywords")
def keywords_endpoint(request: TextRequest):
    keywords, sentences = extract_keywords(request.text, top_n=request.top_n)
    return {
        "keywords": keywords,
        "relevant_sentences": sentences
    }

# Route: Full grammar correction
@app.post("/grammar")
def grammar_endpoint(request: TextRequest):
    corrected = grammar_correct(request.text)
    return {"corrected": corrected}
