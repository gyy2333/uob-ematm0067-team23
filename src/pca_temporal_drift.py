import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import os

def main():

    # load dataset
    df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

    print("Dataset shape:", df.shape)

    # TF-IDF representation
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000
    )

    X = vectorizer.fit_transform(df["abstract"].astype(str))

    print("TF-IDF matrix:", X.shape)

    # PCA reduction
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    df["pca1"] = X_reduced[:,0]
    df["pca2"] = X_reduced[:,1]

    # ensure output directory exists
    os.makedirs("outputs/figures", exist_ok=True)

    # plot
    plt.figure(figsize=(10,7))

    for period in df["period"].unique():
        subset = df[df["period"] == period]

        plt.scatter(
            subset["pca1"],
            subset["pca2"],
            label=period,
            alpha=0.4,
            s=8
        )

    plt.title("Temporal Drift of AI Research (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    plt.tight_layout()

    plt.savefig("outputs/figures/pca_temporal_drift.png", dpi=300)

    print("Saved figure: outputs/figures/pca_temporal_drift.png")


if __name__ == "__main__":
    main()
