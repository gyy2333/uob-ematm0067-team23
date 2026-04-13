import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer


## Global Configurations
# stopwords
STOPWORDS = {
    # 1. Paper Structure
    "paper",
    "article",
    "abstract",
    "introduction",
    "conclusion",
    "section",
    "figure",
    "table",
    "reference",
    "author",
    "authors",
    "et al",
    # 2. Core Concepts
    "model",
    "learning",
    "algorithm",
    "data",
    "dataset",
    "datasets",
    "data set",
    "embeddings",
    "word",
    "words",
    "language",
    "text",
    "feature",
    "features",
    "information",
    # 3. Methodology
    "approach",
    "method",
    "proposed method",
    "system",
    "framework",
    "design",
    "structure",
    "theory",
    "type",
    "types",
    "process",
    "analysis",
    # 4. Context
    "result",
    "results",
    "task",
    "performance",
    "problem",
    "challenge",
    "study",
    "research",
    "work",
    "context",
    "concept",
    "concepts",
    "state art",
    "state-art",
    "state of the art",
    "state",
    "art",
    "discourse",
    "future",
    "learned",
    "effectiveness",
    # 5. Common Verbs
    "propose",
    "proposed",
    "using",
    "used",
    "use",
    "based",
    "identify",
    "achieve",
    "achieves",
    "show",
    "provide",
    "present",
    # 6. Evaluative & Descriptive Modifiers
    "efficient",
    "effective",
    "novel",
    "simple",
    "general",
    "total",
    "high",
    "low",
    "highly",
    "easily",
    "significantly",
    "significant",
    "new",
    "challenging",
    "extensive",
    "different",
    "fine",
    "popular",
    # 7. Connectors, Pronouns & Quantifiers
    "well",
    "various",
    "given",
    "within",
    "however",
    "also",
    "across",
    "one",
    "two",
    "many",
    "set",
    "time",
    "long",
    "futhermore",
    "moreover",
    # 8. Math & Formatting Fragments
    "math",
    "mathsf",
}


class ArxivUltimateVisualizer:
    """
    A visualizer class
    """

    def __init__(
        self,
        data_dir: str = "outputs/pre_processed/",
        plot_dir: str = "outputs/visualization/",
    ) -> None:
        """
        Initializes the visualizer with directories and visual constants

        Args:
            data_dir: Path to the data directory。
            plot_dir: Path to the output plot directory
        """
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        # Time periods for longitudinal analysis
        self.period_order: List[str] = [
            "1991_1998",
            "1999_2006",
            "2007_2014",
            "2015_2021",
        ]

        # Color palette configuration
        self.color_palette: List[str] = ["#2D3436", "#0984E3", "#00B894", "#E17055"]
        self.period_colors: Dict[str, str] = dict(
            zip(self.period_order, self.color_palette)
        )
        self.cat_colors: Dict[str, str] = {
            "cs.AI": "#00B894",
            "cs.LG": "#F1C40F",
            "cs.CL": "#E17055",
        }

        self._setup_style()
        self.data: Dict[str, Any] = {}

    def _setup_style(self) -> None:
        """
        Configures global matplotlib and seaborn styles
        """
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "figure.dpi": 200,
                "figure.facecolor": "white",
                "axes.labelweight": "bold",
            }
        )

    def load_all_data(self) -> bool:
        """
        Loads all required datasets for visualization

        Returns:
            bool: True if loading succeeds, False otherwise
        """
        print("Loading data resources...")
        try:
            self.data["df"] = pd.read_csv(
                f"{self.data_dir}/processed_dataset_final.csv",
                dtype={0: str},
                low_memory=False,
            )
            self.data["embed"] = np.load(f"{self.data_dir}/embeddings.npy")
            self.data["tfidf"] = sparse.load_npz(f"{self.data_dir}/tfidf_matrix.npz")
            self.data["ngram"] = sparse.load_npz(f"{self.data_dir}/ngram_matrix.npz")
            self.data["sim"] = pd.read_csv(
                f"{self.data_dir}/similarity_matrix.csv", index_col=0
            )
            self.data["df_cat"] = pd.read_csv(
                f"{self.data_dir}/keywords_by_category.csv"
            )
            self.data["df_per"] = pd.read_csv(f"{self.data_dir}/keywords_by_period.csv")
            vocab_path = f"{self.data_dir}/tfidf_vocab.pkl"
            self.data["vocab"] = (
                pickle.load(open(vocab_path, "rb"))
                if os.path.exists(vocab_path)
                else None
            )

            print("Resources loaded")
            return True

        except Exception as e:
            print(f"Loading failed: {e}")
            return False

    # Plot

    def plot_1_growth(self) -> None:
        """
        Generates a stacked area chart for publication growth

        Image Purpose: Visualizes publication volume trends across AI subfields
        over time, illustrating macro-level disciplinary expansion and heat.
        """
        plt.figure(figsize=(12, 6))

        growth = self.data["df"].groupby(["year", "categories"]).size()
        growth = growth.unstack().fillna(0)

        colors = [self.cat_colors.get(c, "#BDC3C7") for c in growth.columns]
        growth.plot(kind="area", stacked=True, color=colors, alpha=0.8, ax=plt.gca())

        plt.title("ArXiv AI Subfields Publication Growth (1991-2021)")
        plt.savefig(f"{self.plot_dir}/01_category_growth.png")
        plt.close()

    def plot_2_cloud(self) -> None:
        """
        Generates a deeply denoised word cloud of keywords

        Image Purpose: Highlights the most prominent core academic terms
        """
        # Filter based on aggressive stopwords
        word_freq = {
            str(row["word"]).lower(): row["score"]
            for _, row in self.data["df_per"].iterrows()
            if str(row["word"]).lower() not in STOPWORDS and len(str(row["word"])) > 3
        }

        wc = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            colormap="plasma",
            max_words=60,
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"{self.plot_dir}/02_keywords_period_wordcloud.png")
        plt.close()

    def plot_3_embedding(self) -> None:
        """
        Visualizes semantic clustering using BERT embeddings and t-SNE.

        Image Purpose: identifies semantic clustering and the latent feature space
        distribution of papers across different historical periods.
        """
        # Sample 5000 indices randomly
        idx = np.random.choice(len(self.data["df"]), 5000, replace=False)
        tsne_results = TSNE(
            n_components=2, perplexity=45, random_state=42
        ).fit_transform(self.data["embed"][idx])

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=self.data["df"].iloc[idx]["period"],
            hue_order=self.period_order,
            palette=self.period_colors,
            s=30,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(
            f"{self.plot_dir}/03_embeddings_clustering.png", bbox_inches="tight"
        )
        plt.close()

    def plot_4_trajectory(self) -> None:
        """
        Plots the macroscopic semantic trajectory across eras.

        Image Purpose: Tracks the overall macroscopic paradigm shift of AI research
        across multiple eras utilizing SVD on TF-IDF matrices.
        """
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(self.data["tfidf"])

        def get_axis_labels(comp: np.ndarray) -> Tuple[str, str]:
            """
            Helper to get semantic meaning of SVD axes
            """
            if self.data["vocab"] is None:
                return "Dim Low", "Dim High"

            idx = comp.argsort()
            valid = lambda w: str(w).lower() not in STOPWORDS

            low_terms = [
                self.data["vocab"][i] for i in idx if valid(self.data["vocab"][i])
            ]
            high_terms = [
                self.data["vocab"][i] for i in idx[::-1] if valid(self.data["vocab"][i])
            ]
            return low_terms[0], high_terms[0]

        x_labels, y_labels = get_axis_labels(svd.components_[0]), get_axis_labels(
            svd.components_[1]
        )

        plt.figure(figsize=(12, 8))

        # Plot scatter
        for period in self.period_order:
            mask = self.data["df"]["period"] == period
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=self.period_colors[period],
                s=2,
                alpha=0.1,
            )

        # Calculate and plot the trajectory path
        res = pd.DataFrame(coords, columns=["x", "y"])
        res = res.assign(p=self.data["df"]["period"])
        res = res.groupby("p").mean().reindex(self.period_order)

        plt.plot(res["x"], res["y"], color="#2d3436", lw=2, ls="--", alpha=0.5)
        for period in res.index:
            plt.scatter(
                res.loc[period, "x"],
                res.loc[period, "y"],
                s=450,
                color=self.period_colors[period],
                edgecolors="white",
                lw=3,
                label=period,
                zorder=12,
            )

        plt.xlabel(f"← {x_labels[0]} | Semantic Shift | {x_labels[1]} →")
        plt.ylabel(f"← {y_labels[0]} | Focus Shift | {y_labels[1]} →")
        plt.legend(title="Periods", markerscale=0.6)

        plt.savefig(f"{self.plot_dir}/04_tfidf_trajectory.png")
        plt.close()

    def plot_5_heatmap(self) -> None:
        """
        Plots the global category similarity heatmap

        Image Purpose: Quantifies and displays the similarity matrix, showing
        the closeness and intersections among different research categories
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data["sim"], annot=True, cmap="YlGnBu", fmt=".3f")
        plt.savefig(f"{self.plot_dir}/05_similarity_heatmap.png")
        plt.close()

    def plot_6_ngram_density_complexity(self) -> None:
        """
        Plots a boxplot for term complexity based on N-gram density

        Image Purpose: Assesses academic expression complexity by evaluating
        the density of multi-word phrases over different periods
        """
        df = self.data["df"].copy()
        df["ngram_density"] = np.array((self.data["ngram"] > 0).sum(axis=1)).flatten()

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x="period",
            y="ngram_density",
            order=self.period_order,
            hue="period",
            palette=self.period_colors,
            showfliers=False,
            legend=False,
        )

        plt.title("Academic Expression Complexity (N-gram Density)")
        plt.savefig(f"{self.plot_dir}/06_ngram_density_complexity.png")
        plt.close()

    def plot_7_fingerprint(self) -> None:
        """
        Plots a heatmap fingerprinting technology focus across sub-domains

        Image Purpose: Compares exclusive technical foci across distinct CS fields,
        highlighting their commonalities and individual distinguishing features
        """
        df_cat = self.data["df_cat"].copy()

        # Filter out stopwords
        df_cat = df_cat[~df_cat["word"].str.lower().str.strip().isin(STOPWORDS)]

        # Select top 30 technical terms overall
        top_overall = df_cat.groupby("word")["score"].sum().nlargest(30).index
        pivot_df = (
            df_cat[df_cat["word"].isin(top_overall)]
            .pivot(index="word", columns="category", values="score")
            .fillna(0)
        )

        plt.figure(figsize=(12, 11))
        sns.heatmap(
            pivot_df,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            cbar_kws={"label": "TF-IDF Weight"},
        )
        plt.title(
            "Domain Fingerprint: Technology Focus Across Fields",
            pad=20,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/07_category_fingerprint.png")
        plt.close()

    def plot_8_evolution(self) -> None:
        """
        Plots a heatmap tracking the rise and decay of AI concepts over eras

        Image Purpose: Provides a timeline view of when specific technologies
        gained traction or phased out, representing knowledge transition
        """
        df_per = self.data["df_per"].copy()

        # filtering of noise
        df_per = df_per[~df_per["word"].str.lower().str.strip().isin(STOPWORDS)]

        pivot_per = df_per.pivot(index="word", columns="period", values="score").fillna(
            0
        )
        pivot_per = pivot_per.reindex(columns=self.period_order)

        # Extract the top 10 words for each era to track
        plot_words: List[str] = []
        for period in self.period_order:
            plot_words.extend(pivot_per[period].nlargest(10).index.tolist())

        # Remove duplicates while preserving order
        plot_words = list(dict.fromkeys(plot_words))

        plt.figure(figsize=(12, 15))
        sns.heatmap(pivot_per.loc[plot_words], annot=True, cmap="Spectral_r", fmt=".1f")
        plt.title(
            "Tech Transition: Rise and Decay of AI Concepts", pad=20, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/08_keywords_period_evolution.png")
        plt.close()

    def plot_9_phrases(self) -> None:
        """
        Extracts and plots the top technical phrases for each era

        Image Purpose: Provides a highly interpretable list of key N-grams
        representing the exact breakthroughs of each time period.
        """
        fig, axes = plt.subplots(1, 4, figsize=(26, 8))

        for i, period in enumerate(self.period_order):
            subset = self.data["df"][self.data["df"]["period"] == period][
                "clean_text"
            ].dropna()
            cv = CountVectorizer(
                ngram_range=(2, 3), max_features=50, stop_words="english"
            )

            try:
                counts = cv.fit_transform(subset)
                words = cv.get_feature_names_out()
                freqs = counts.sum(axis=0).A1

                # Sort indices by frequency descending
                idx = freqs.argsort()[::-1]
                filtered_words: List[str] = []
                filtered_freqs: List[int] = []

                for k in idx:
                    w = words[k]
                    # Filter out fragments
                    if not any(j in w for j in STOPWORDS):
                        filtered_words.append(w)
                        filtered_freqs.append(freqs[k])

                # Plot top 10
                sns.barplot(
                    x=filtered_freqs[:10],
                    y=filtered_words[:10],
                    ax=axes[i],
                    hue=filtered_words[:10],
                    palette=[self.period_colors[period]] * 10,
                    legend=False,
                )
                axes[i].set_title(
                    f"Key Terms: {period}",
                    color=self.period_colors[period],
                    fontweight="bold",
                )

            except Exception as e:
                print(f"Warning: Failed to generate phrases for {period}. Reason: {e}")

        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/09_multi_phrases.png")
        plt.close()


def main():
    viz = ArxivUltimateVisualizer()
    if viz.load_all_data():
        viz.plot_1_growth()
        viz.plot_2_cloud()
        viz.plot_3_embedding()
        viz.plot_4_trajectory()
        viz.plot_5_heatmap()
        viz.plot_6_ngram_density_complexity()
        viz.plot_7_fingerprint()
        viz.plot_8_evolution()
        viz.plot_9_phrases()
        print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
