from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

MODEL_PATH = "models/oku_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def translate_oku_to_english(text):

    prompt = f"translate Oku to English: {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

def translate_english_to_oku(text):

    prompt = f"translate English to Oku: {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

if __name__ == "__main__":

    direction = input(
        "1 = Oku->English\n2 = English->Oku\nChoice: "
    )

    sentence = input("Enter text: ")

    if direction == "1":
        print(translate_oku_to_english(sentence))
    else:
        print(translate_english_to_oku(sentence))