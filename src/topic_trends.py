import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt


df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

vectorizer = CountVectorizer(stop_words="english", max_features=2000)
X = vectorizer.fit_transform(df["abstract"].astype(str))

lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(X)

df["dominant_topic"] = topics.argmax(axis=1)

topic_counts = pd.crosstab(df["period"], df["dominant_topic"])

os.makedirs("outputs/figures", exist_ok=True)

topic_counts.plot(kind="bar", stacked=True, figsize=(10,6))

plt.title("Topic Distribution Across Time Periods")
plt.xlabel("Time Period")
plt.ylabel("Number of Papers")

plt.tight_layout()

plt.savefig("outputs/figures/topic_trends.png")

print("Saved figure: outputs/figures/topic_trends.png")
