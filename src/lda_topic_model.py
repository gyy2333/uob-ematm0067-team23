import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def main():

    df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=2000
    )

    X = vectorizer.fit_transform(df["abstract"].astype(str))

    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42
    )

    lda.fit(X)

    words = vectorizer.get_feature_names_out()

    topics = []

    for topic_idx, topic in enumerate(lda.components_):

        top_indices = topic.argsort()[-10:][::-1]

        keywords = [words[i] for i in top_indices]

        topics.append({
            "topic": topic_idx,
            "keywords": ", ".join(keywords)
        })

    topics_df = pd.DataFrame(topics)

    os.makedirs("outputs/tables", exist_ok=True)

    topics_df.to_csv("outputs/tables/lda_topics.csv", index=False)

    print("\nLDA Topics:")
    print(topics_df)

    print("\nSaved to outputs/tables/lda_topics.csv")


if __name__ == "__main__":
    main()
