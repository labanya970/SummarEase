from transformers import pipeline

# Define this BEFORE the function
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if not text.strip():
        return "⚠️ Input text is empty. Please provide valid content."

    text = " ".join(text.split()[:1024])  # Limit token count
    num_tokens = len(text.split())

    min_length = max(40, num_tokens // 2)
    max_length = min(512, int(num_tokens * 0.9))

    if min_length >= max_length:
        min_length = max_length - 10 if max_length > 10 else max_length

    try:
        summary = bart_summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            temperature=1.2,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.05,
            early_stopping=True
        )
        return summary[0].get('summary_text', "⚠️ Unable to generate summary.")
    except Exception as e:
        return f"⚠️ An error occurred during summarization: {str(e)}"
