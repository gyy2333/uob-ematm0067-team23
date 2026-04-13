from .modeling import Modeling
from .modeling import COMBINED_STOPWORDS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDA(Modeling):
    def __init__(self, data_path, output_path):
        super().__init__(data_path, output_path)
        print("Loading input data:", self.data_path)
        self.output_path = self.output_path + "lda/"
        os.makedirs(self.output_path, exist_ok=True)

        self.vectorizer = None
        self.lda_model = None
        self._x_counts = None
        self._topic_distributions = None
        self._feature_names = None

    def build_model(
        self, 
        n_topics=5, 
        max_df=0.8, 
        min_df=5, 
        max_features=1000, 
        ngram_range=(1, 2)
    ):
        """Initialize the Vectorizer and LDA model with predefined parameters."""
        self.vectorizer = CountVectorizer(
            stop_words=list(COMBINED_STOPWORDS),
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=42,              
            learning_method='online'
        )
        return self.lda_model

    def train_model(self, text_col="clean_text"):
        """
        Vectorize text data and fit the LDA model
        """
        print("Training LDA model...")
        
        # Ensure target columns exist and drop missing rows
        self._df = self._df.dropna(subset=["year", text_col]).copy()
        self._df["year"] = self._df["year"].astype(int)
        texts = self._df[text_col].astype(str)

        # 1. Fit Transform Vectorizer
        self._x_counts = self.vectorizer.fit_transform(texts)
        self._feature_names = self.vectorizer.get_feature_names_out().tolist()

        # 2. Fit Transform LDA
        self._topic_distributions = self.lda_model.fit_transform(self._x_counts)
        print("LDA training completed.")

    def save_output(self, n_top_words=10):
        """
        Save keyword distributions and document/year topic distributions to CSV
        """
        print("Saving LDA outputs...")
        
        # 1. Save top words per topic
        topic_words = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [self._feature_names[i] for i in top_features_ind]
            topic_words.append(
                {
                    "Topic": f"Topic {topic_idx + 1}",
                    "Keywords": ", ".join(top_features),
                }
            )
        pd.DataFrame(topic_words).to_csv(
            self.output_path + "lda_topic_keywords.csv", index=False
        )

        # 2. Save topic evolution by year
        topic_cols = [f"Topic {i+1}" for i in range(self.lda_model.n_components)]
        topic_dist_df = pd.DataFrame(
            self._topic_distributions, index=self._df.index, columns=topic_cols
        )

        topic_evo = (
            pd.concat([self._df[["year"]], topic_dist_df], axis=1)
            .groupby("year")
            .mean()
        )
        topic_evo.to_csv(self.output_path + "lda_topic_evolution_by_year.csv")

    def visualization(self):
        """
        Generate and save the Stackplot for LDA Topic Evolution
        """
        print("Generating LDA Topic Evolution visualization...")
        
        THEME_COLORS = ["#00798C", "#D1495B", "#30638E", "#EDAE49", "#66A182"]
        N_TOPICS = self.lda_model.n_components

        # Prepare evolution data
        topic_cols = [f"Topic {i+1}" for i in range(N_TOPICS)]
        topic_dist_df = pd.DataFrame(
            self._topic_distributions, index=self._df.index, columns=topic_cols
        )
        topic_evo = pd.concat([self._df[["year"]], topic_dist_df], axis=1).groupby("year").mean()

        # Extract keywords
        topic_words = [
            [str(self._feature_names[i]) for i in topic.argsort()[:-11:-1]]
            for topic in self.lda_model.components_
        ]

        all_words_flat = (w for t in topic_words for w in t)
        repeated = {w for w, c in Counter(all_words_flat).items() if c > 1}

        content_lines = []
        for i, words in enumerate(topic_words):
            fmt = [f"* {w.upper()}" if w in repeated else w for w in words]
            content_lines.append(f"Topic {i+1}: {', '.join(fmt)}")

        milestones = {}
        if "period" in self._df.columns:
            milestones = {
                year: str(name).replace('_', '-')  
                for name, year in self._df.groupby("period")["year"].min().items()
            }

        # Plot Initialization
        fig, ax = plt.subplots(figsize=(14, 9))
        plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.28)

        colors = THEME_COLORS * (N_TOPICS // len(THEME_COLORS) + 1)

        ax.stackplot(
            topic_evo.index,
            topic_evo.T,
            labels=topic_evo.columns,
            colors=colors[:N_TOPICS],
            alpha=0.85,
        )

        y_limit_upper = float(ax.get_ylim()[1])
        for yr, lbl in milestones.items():
            if yr in topic_evo.index:
                ax.axvline(x=yr, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
                ax.text(
                    yr + 0.1,
                    y_limit_upper * 0.95,
                    lbl,
                    va="top",
                    ha="left",
                    fontsize=11,
                    fontweight="bold",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
                )

        ax.set_title("LDA Topic Evolution", pad=35, fontweight="bold", fontsize=14)
        ax.set(
            xlabel="Year",
            ylabel="Topic Intensity",
            xticks=sorted(self._df["year"].unique()),
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            bbox_to_anchor=(0.5, 1.02),
            loc="lower center",
            ncol=N_TOPICS,
            frameon=False,
        )

        # Keyword Info Panel
        fig.text(
            0.5, 0.12, "\n" * 7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1.5", fc="whitesmoke", ec="lightgray", alpha=0.8),
        )
        fig.text(
            0.5, 0.19, "| Core Keywords per Topic (Top 10) |",
            ha="center", va="center", fontweight="bold",
        )
        fig.text(
            0.5, 0.17, "* UPPERCASE = Overlapping words across topics",
            ha="center", va="center", fontsize=10, alpha=0.7,
        )
        fig.text(
            0.08, 0.11, "\n".join(content_lines),
            ha="left", va="center", fontsize=11,
        )

        # Save Image
        save_path = os.path.join(self.output_path, "lda_topic_evolution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {save_path}")

    def process(self):
        """
        Orchestrate the LDA pipeline
        """
        print("\n=== Initializing LDA Pipeline ===")
        self.load_data()
        self.build_model(n_topics=5)
        self.train_model(text_col="clean_text")
        self.save_output()
        self.visualization()
        print("=== LDA Pipeline Finished ===\n")