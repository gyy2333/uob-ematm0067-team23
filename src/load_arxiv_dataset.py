import json
import pandas as pd

print("Loading dataset...")

papers = []

with open("data/raw/arxiv-abstracts.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 20000:
            break

        paper = json.loads(line)

        papers.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "categories": paper.get("categories"),
            "update_date": paper.get("update_date")
        })

df = pd.DataFrame(papers)

df.to_csv("data/raw/arxiv_subset.csv", index=False)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
