import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/Oku-English_cleaned.csv")

df = df.dropna()
df = df.drop_duplicates()

df = df.rename(columns={
    "Oku_Text": "source",
    "English_NIV_Equivalent": "target"
})

expanded_data = []

for _, row in df.iterrows():

    oku = str(row["source"]).strip()
    eng = str(row["target"]).strip()

    if len(oku) > 0 and len(eng) > 0:

        expanded_data.append({
            "source": f"translate Oku to English: {oku}",
            "target": eng
        })

        expanded_data.append({
            "source": f"translate English to Oku: {eng}",
            "target": oku
        })

df = pd.DataFrame(expanded_data)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Dataset prepared successfully")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")