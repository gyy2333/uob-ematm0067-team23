import json
import pandas as pd

TARGET_CATS = {"cs.AI", "cs.LG", "cs.CL", "stat.ML"}

def parse_year_from_id(arxiv_id):
    """
    Parse year from arXiv ID.
    New-style IDs look like 0704.0001 -> 2007
    We map 07-99 -> 2007-2099, but in practice this dataset is modern.
    """
    try:
        prefix = str(arxiv_id).split(".")[0]
        yy = int(prefix[:2])
        if yy >= 7:
            return 2000 + yy
        return None
    except Exception:
        return None

def normalize_categories(cat_value):
    if isinstance(cat_value, list):
        return cat_value
    if isinstance(cat_value, str):
        text = cat_value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return text.replace(",", " ").split()
    return []

def main():
    papers = []

    with open("data/raw/arxiv-abstracts.jsonl", "r") as f:
        for i, line in enumerate(f):
            paper = json.loads(line)

            categories = normalize_categories(paper.get("categories"))
            if not any(cat in TARGET_CATS for cat in categories):
                continue

            year = parse_year_from_id(paper.get("id"))
            if year is None:
                continue

            abstract = paper.get("abstract")
            title = paper.get("title")

            if not abstract or not title:
                continue

            papers.append({
                "id": paper.get("id"),
                "title": title,
                "abstract": abstract,
                "categories": " ".join(categories),
                "year": year
            })

            if len(papers) % 5000 == 0:
                print(f"Collected {len(papers)} matching papers...")

    df = pd.DataFrame(papers)

    print("\nFinal shape:", df.shape)
    print("Year range:", df["year"].min(), "to", df["year"].max())
    print("\nTop categories:")
    print(df["categories"].value_counts().head(10))

    df.to_csv("data/processed/ai_ml_nlp_full_filtered.csv", index=False)
    print("\nSaved to data/processed/ai_ml_nlp_full_filtered.csv")

if __name__ == "__main__":
    main()
