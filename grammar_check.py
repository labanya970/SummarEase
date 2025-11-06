from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def grammar_correct(text: str) -> str:
    """General-purpose grammar correction without changing meaning."""
    input_text = "fix grammar: " + text.strip()
    inputs = tokenizer([input_text], return_tensors="pt", truncation=False)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=5,
        early_stopping=True
    )
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected


if __name__ == "__main__":
    paragraph = """
    Yesterday, I am walking to the store when it suddenly start to rain.
    I forget to bring my umbrella, so I runs back home as fast as I can.
    On the way, I see my neighbor who were trying to cover her groceries with a newspaper.
    We both laughs at how unlucky we are.
    When I finally reach home, my shoes was completely wet and my phone doesn’t work anymore.
    I wish I would have checked the weather before leaving.
    """

    print("📝 Original Text:")
    print(paragraph)

    corrected = grammar_correct(paragraph)

    print("\n✅ Grammar Corrected:")
    print(corrected)
