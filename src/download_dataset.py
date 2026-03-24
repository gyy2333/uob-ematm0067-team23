from datasets import load_dataset
import pandas as pd
from pathlib import Path

def main():
    print("Downloading arXiv dataset subset...")

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "ccdv/arxiv-classification",
        split="train[:10000]"
    )

    df = pd.DataFrame(dataset)

    # rename text column to abstract for consistency
    df = df.rename(columns={"text": "abstract"})

    # keep only useful columns
    df = df[["abstract", "label"]]

    df.to_csv(out_dir / "arxiv_abstracts_subset.csv", index=False)

    print("\nSaved cleaned subset to data/raw/arxiv_abstracts_subset.csv")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
