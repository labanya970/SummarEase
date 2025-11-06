import os
import torch
import re
import nltk
import language_tool_python
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize

nltk.data.path.append("C:/Users/LABANYA THAKUR/nltk_data")

class Config:
    MODEL_NAME = "t5-small"
    SAVE_PATH = "grammar_enhanced_summarizer.pth"
    DATASET_NAME = "cnn_dailymail"
    DATASET_VERSION = "3.0.0"
    CACHE_DIR = "enhanced_data_cache"
    MAX_INPUT_LEN = 512
    MAX_OUTPUT_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    GEN_MAX_LENGTH = 150
    GEN_MIN_LENGTH = 50
    TEMPERATURE = 2.6
    TOP_P = 0.95
    REPETITION_PENALTY = 2.0
    LENGTH_PENALTY = 0.7
    NO_REPEAT_NGRAM_SIZE = 3
    NUM_BEAMS = 4
    NUM_RETURN_SEQUENCES = 4
    DO_SAMPLE = True
    MAX_GRAMMAR_ERRORS = 1
    GRAMMAR_LANGUAGE = 'en-US'
    SIMILARITY_WEIGHT = 0.3
    CREATIVITY_WEIGHT = 0.7
    MIN_SUMMARY_WORDS = 25
    MIN_SENTENCE_LENGTH = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME, legacy=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
refiner = pipeline("text2text-generation", model="google/flan-t5-base")

try:
    tool = language_tool_python.LanguageTool(Config.GRAMMAR_LANGUAGE)
    print("✅ Grammar checker initialized")
except Exception as e:
    print(f"⚠ Failed to initialize grammar checker: {e}")
    tool = None

def check_and_correct_grammar(text):
    if tool is None:
        return text
    try:
        matches = tool.check(text)
        if len(matches) > Config.MAX_GRAMMAR_ERRORS:
            return tool.correct(text)
        return text
    except:
        return text

def clean_summary(summary):
    summary = summary.strip()
    summary = re.sub(r"\s+([.,!?;])", r"\1", summary)
    sentences = sent_tokenize(summary)
    valid = [s for s in sentences if len(s.split()) >= Config.MIN_SENTENCE_LENGTH]
    return " ".join(valid)

def score_summary(input_text, candidate):
    sim_score = float(util.cos_sim(
        embedder.encode(input_text, convert_to_tensor=True),
        embedder.encode(candidate, convert_to_tensor=True)
    ))
    words = candidate.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if not candidate.strip().endswith(('.', '?', '!')):
        sim_score -= 0.1
    return Config.SIMILARITY_WEIGHT * sim_score + Config.CREATIVITY_WEIGHT * unique_ratio

def refine_summary(summary_text):
    prompt = ("Refine the following summary to make it fluent, semantically consistent, "
              "and slightly more abstract, without losing meaning:\n" + summary_text)
    refined = refiner(prompt, max_length=200, do_sample=False)
    return refined[0]['generated_text']

def generate_quality_summary(text):
    global model
    model.eval()
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=Config.MAX_INPUT_LEN,
        truncation=True,
        padding="max_length"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=Config.GEN_MAX_LENGTH,
            min_length=Config.GEN_MIN_LENGTH,
            do_sample=Config.DO_SAMPLE,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            num_beams=Config.NUM_BEAMS,
            num_return_sequences=Config.NUM_RETURN_SEQUENCES,
            no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=Config.REPETITION_PENALTY,
            length_penalty=Config.LENGTH_PENALTY,
            early_stopping=True
        )
    decoded = [tokenizer.decode(s, skip_special_tokens=True) for s in outputs]
    corrected = [check_and_correct_grammar(s) for s in decoded]
    cleaned = [clean_summary(c) for c in corrected]
    scored = [(score_summary(text, c), c) for c in cleaned if len(c.split()) >= Config.MIN_SUMMARY_WORDS]
    if not scored:
        return "⚠️ No valid summaries generated. Try tweaking parameters or input text."
    best_summary = max(scored, key=lambda x: x[0])[1]
    refined = refine_summary(best_summary)
    return refined

def train_model():
    dataset = load_dataset(Config.DATASET_NAME, Config.DATASET_VERSION, split="train[:5000]")
    def preprocess(examples):
        inputs = tokenizer(
            ["summarize: " + text for text in examples["article"]],
            max_length=Config.MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        outputs = tokenizer(
            examples["highlights"],
            max_length=Config.MAX_OUTPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": outputs["input_ids"]
        }
    processed = dataset.map(preprocess, batched=True, remove_columns=["article", "highlights", "id"])
    processed.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    loader = DataLoader(processed, batch_size=Config.BATCH_SIZE, shuffle=True)
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    model.train()
    for epoch in range(Config.EPOCHS):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch in progress:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=loss.item())
    torch.save(model.state_dict(), Config.SAVE_PATH)
    return model.eval()

model = None
if os.path.exists(Config.SAVE_PATH):
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME).to(device)
    model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded from saved weights.")
else:
    print("🔄 No saved model found. Training from scratch...")
    model = train_model()

if __name__ == "__main__":
    input_text = """
    Artificial intelligence, commonly known as AI, is a branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. 
    These tasks include understanding language, recognizing images, making decisions, and even generating creative content. 
    AI works through advanced algorithms and machine learning models that learn patterns from large amounts of data and improve over time without explicit programming. 
    It is used in everyday applications such as voice assistants, recommendation systems, self-driving cars, and medical diagnostics. 
    The impact of AI on industries is immense — it increases efficiency, reduces human error, and enables innovation in fields like healthcare, education, and finance. 
    However, as AI becomes more powerful, ethical concerns such as data privacy, job displacement, and algorithmic bias have also emerged. 
    Balancing technological progress with responsible use of AI is crucial to ensure that it benefits society as a whole.
   
    """
    print("Original Paragraph:\n", input_text.strip())
    summary = generate_quality_summary(input_text)
    print("\nFinal Grammar-Coherent Abstractive Summary:\n", summary)
