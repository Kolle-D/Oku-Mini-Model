import pandas as pd
import sacrebleu

from inference import (
    translate_oku_to_english,
    translate_english_to_oku
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

predictions = []
references = []

for _, row in test_df.iterrows():

    source = row["source"]

    if source.startswith(
        "translate Oku to English:"
    ):

        text = source.replace(
            "translate Oku to English:",
            ""
        ).strip()

        pred = translate_oku_to_english(text)

    else:

        text = source.replace(
            "translate English to Oku:",
            ""
        ).strip()

        pred = translate_english_to_oku(text)

    predictions.append(pred)

    references.append(row["target"])

bleu = sacrebleu.corpus_bleu(
    predictions,
    [references]
)

print(
    f"\nBLEU Score: {bleu.score:.2f}"
)