import re
import pandas as pd

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

df["clean_abstract"] = df["abstract"].apply(clean_text)
df["abstract_length"] = df["clean_abstract"].str.split().apply(len)

# remove very short abstracts
df = df[df["abstract_length"] >= 30].copy()

df.to_csv("data/processed/ai_ml_nlp_clean.csv", index=False)

print("Cleaned shape:", df.shape)
print("Year range:", df["year"].min(), "to", df["year"].max())
print("\nAbstract length summary:")
print(df["abstract_length"].describe())
print("\nSample rows:")
print(df[["year", "categories", "abstract_length"]].head())
