## loading the data after the data undergoes EDA
import pandas as pd
import re
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import save_npz

from sentence_transformers import SentenceTransformer

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# LOAD DATA
def load_data(path):
    df = pd.read_csv(path)
    print("Dataset shape", df.shape)
    print(df.info())
    print(type(df))
    return df


def remove_special(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def preprocess_text(df):
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def to_wordnet_pos(treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        if treebank_tag.startswith("V"):
            return wordnet.VERB
        if treebank_tag.startswith("N"):
            return wordnet.NOUN
        if treebank_tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def pipeline(text):
        # Step 1: conerting to Lower case
        text = text.lower()
        # Step 2: Removing the special characters
        text = remove_special(text)
        # Step 3: Tokenization
        words = text.split()
        # Step 4: Remove stopwords
        words = [word for word in words if word not in stop_words]
        # Step 5: POS tagging + lemmatization
        tagged_words = pos_tag(words)
        words = [
            lemmatizer.lemmatize(word, to_wordnet_pos(tag))
            for word, tag in tagged_words
        ]
        # Step 6: Join back to string
        return " ".join(words)

    df["clean_text"] = df["abstract"].apply(pipeline)
    print("Sample cleaned text", df["clean_text"].iloc[100])
    df.to_csv("../data/modeling/ai_ml_nlp_dataset_labeled.csv", index=False)
    return df


## TF-IDF
def tfidf_representation(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_text"])

    save_npz("../data/pre_processed/tfidf_matrix.npz", X)
    print("TF-IDF Matrix saved!")

    with open("../data/pre_processed/tfidf_vocab.pkl", "wb") as f:
        pickle.dump(tfidf.vocabulary_, f)

    print("TF_IDF vocabulary saved")

    return tfidf, X


##analysis
##Top keywords


def top_keywords(tfidf, X, top_n=100):
    feature_names = tfidf.get_feature_names_out()
    sums = np.array(X.sum(axis=0)).flatten()

    top_indices = sums.argsort()[-550:]
    top_words = [feature_names[i] for i in top_indices]

    keywords_df = pd.DataFrame({"word": top_words})

    keywords_df.to_csv("../data/pre_processed/top_keywords.csv", index=False)
    print("Top keywords saved!")

    print(len(top_words))
    print("\nTop Keywords:\n", top_words)


## by time period


def keywords_by_period(df):
    tfidf = TfidfVectorizer(max_features=5000)

    rows = []

    for period in df["period"].unique():
        subset = df[df["period"] == period]

        X_p = tfidf.fit_transform(subset["clean_text"])
        features = tfidf.get_feature_names_out()

        sums = np.array(X_p.sum(axis=0)).flatten()
        top_idx = sums.argsort()[-500:]

        for i in top_idx:
            rows.append({"period": period, "word": features[i], "score": sums[i]})

    pd.DataFrame(rows).to_csv(
        "../data/pre_processed/keywords_by_period.csv", index=False
    )

    print("Keywords by periods saved")


## by category
def keywords_by_category(df):
    tfidf = TfidfVectorizer(max_features=5000)

    rows = []

    for cat in df["categories"].unique():
        subset = df[df["categories"] == cat]

        X_c = tfidf.fit_transform(subset["clean_text"])
        features = tfidf.get_feature_names_out()

        sums = np.array(X_c.sum(axis=0)).flatten()
        top_idx = sums.argsort()[-500:]

        for i in top_idx:
            rows.append({"category": cat, "word": features[i], "score": sums[i]})

    pd.DataFrame(rows).to_csv(
        "../data/pre_processed/keywords_by_category.csv", index=False
    )

    print("Keywords by category saved")


# N- grams
# using the combination of bi-grams and tri-grams
def ngram_representation(df):
    tfidf_ngram = TfidfVectorizer(ngram_range=(2, 3), max_features=5000)
    X_ngram = tfidf_ngram.fit_transform(df["clean_text"])
    print("N-gram matrix shape:", X_ngram.shape)

    save_npz("../data/pre_processed/ngram_matrix.npz", X_ngram)

    print("N-grams saved")

    return X_ngram


## embedding


def generate_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_embed = model.encode(df["clean_text"], show_progress_bar=True)

    np.save("../data/pre_processed/embaddings.npy", X_embed)

    print("Embeddings saved")

    return X_embed


## compare similarity across periods
##Elements in the same period (row) show trends rather than direct similarity


def similarity_analysis(df, X_embed):
    period_groups = df.groupby("period")

    period_vectors = {}
    for period, group in period_groups:
        period_vectors[period] = np.mean(X_embed[group.index], axis=0)

    periods = sorted(list(period_vectors.keys()))
    sim_matrix = []

    print("\nSimilarity Between Periods:")

    for p1 in periods:
        row = []
        for p2 in periods:
            sim = cosine_similarity([period_vectors[p1]], [period_vectors[p2]])[0][0]
            row.append(sim)
        sim_matrix.append(row)

    sim_df = pd.DataFrame(sim_matrix)
    sim_df.index = periods
    sim_df.columns = periods

    sim_df.to_csv("../data/pre_processed/similarity_matrix.csv")

    print("Similarity mareix saved")
    print(sim_df)


## Classification - Logistic Regression
def classification(df):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["clean_text"])
    y = df["period"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)


def main():
    path = "../data/raw_data/ai_ml_nlp_datasetv11.csv"

    df = load_data(path)
    df = preprocess_text(df)

    # tfidf, X = tfidf_representation(df)
    # top_keywords(tfidf, X)
    # keywords_by_period(df)
    # keywords_by_category(df)

    # ngram_representation(df)

    # X_embed = generate_embeddings(df)
    # similarity_analysis(df, X_embed)

    # classification(df)


if __name__ == "__main__":
    main()
