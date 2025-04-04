import spacy
from transformers import pipeline
import language_tool_python

def check_grammar(text):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)
    corrected_text = text
    for match in reversed(matches):
        if match.replacements:
            corrected_text = corrected_text[:match.offset] + match.replacements[0] + corrected_text[match.offset + match.errorLength:]
    return corrected_text

def ensure_tense_consistency(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    corrected_tokens = []
    past_tense = False
    for token in doc:
        if token.pos_ == "VERB" and "Tense=Past" in token.morph:
            past_tense = True
        corrected_tokens.append(token.text)
    if past_tense:
        corrected_tokens = [token.lemma_ if token.pos_ == "VERB" else token.text for token in doc]
    return " ".join(corrected_tokens)

def correct_grammar_using_ai(text):
    grammar_corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    corrected_output = grammar_corrector(text)[0]['generated_text']
    return corrected_output

def full_grammar_fix(text):
    text = check_grammar(text)
    text = ensure_tense_consistency(text)
    text = correct_grammar_using_ai(text)
    return text


