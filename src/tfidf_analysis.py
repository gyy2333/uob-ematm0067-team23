import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms_by_period(df, text_col, period_col, ngram_range=(1,1), top_k=20):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(df[text_col])
    terms = vectorizer.get_feature_names_out()

    rows = []

    for period in df[period_col].unique():

        idx = (df[period_col] == period).values
        mean_scores = np.asarray(X[idx].mean(axis=0)).ravel()

        top_ids = mean_scores.argsort()[-top_k:][::-1]

        for i in top_ids:
            rows.append({
                "period": period,
                "term": terms[i],
                "score": mean_scores[i]
            })

    return pd.DataFrame(rows)


def main():

    df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

    unigram_df = top_terms_by_period(
        df=df,
        text_col="abstract",
        period_col="period",
        ngram_range=(1,1),
        top_k=20
    )

    unigram_df.to_csv("outputs/tables/top_unigrams_by_period.csv", index=False)

    bigram_df = top_terms_by_period(
        df=df,
        text_col="abstract",
        period_col="period",
        ngram_range=(2,2),
        top_k=20
    )

    bigram_df.to_csv("outputs/tables/top_bigrams_by_period.csv", index=False)

    print("Saved TF-IDF results.")


if __name__ == "__main__":
    main()
