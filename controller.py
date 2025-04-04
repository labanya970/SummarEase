from fastapi import FastAPI, Request
from pydantic import BaseModel
from summarize_model import summarize_text
from highlight import extract_keywords, extract_sentences_with_keywords
from grammar_check import full_grammar_fix
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TextRequest(BaseModel):
    text: str
    top_n: int = 10

@app.post("/summarize")
def summarize_endpoint(request: TextRequest):
    return {"summary": summarize_text(request.text)}

@app.post("/keywords")
def keywords_endpoint(request: TextRequest):
    keywords, sentences = extract_keywords(request.text, top_n=request.top_n)
    return {"keywords": keywords, "relevant_sentences": sentences}

@app.post("/grammar")
def grammar_endpoint(request: TextRequest):
    return {"corrected": full_grammar_fix(request.text)}
