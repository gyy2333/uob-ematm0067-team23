from .modeling import Modeling
from .modeling import COMBINED_STOPWORDS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        """
        Initialize the Vectorizer and LDA model with predefined parameters
        """
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
        cols_to_keep = ["period", "categories", text_col]
        self._df = self._df.dropna(subset=cols_to_keep).copy()
        
        self._df['categories'] = self._df['categories'].str.split()
        self._df = self._df.explode('categories').reset_index(drop=True)
        
        texts = self._df[text_col].astype(str)
        self._x_counts = self.vectorizer.fit_transform(texts)
        self._feature_names = self.vectorizer.get_feature_names_out().tolist()
        self._topic_distributions = self.lda_model.fit_transform(self._x_counts)
        print(f"LDA completed. Processed {len(self._df)} categorical entries.")

    def save_output(self, n_top_words=10):
        """
        Saves a detailed CSV showing Contextual Keywords for EVERY topic within each slice
        """
        print(f"Saving LDA outputs...")
        
        
        word_topic_dist = self.lda_model.components_ / self.lda_model.components_.sum(axis=1)[:, np.newaxis]
        
        results = []
        for (period, cat), group in self._df.groupby(["period", "categories"]):
            indices = group.index.tolist()
            slice_topic_intensities = self._topic_distributions[indices].mean(axis=0)
            slice_word_counts = self._x_counts[indices].sum(axis=0).A1
            
            for t_idx in range(self.lda_model.n_components):
                # Contextual Score: Global Probability * Local Frequency
                contextual_scores = word_topic_dist[t_idx] * slice_word_counts
                top_indices = contextual_scores.argsort()[-n_top_words:][::-1]
                topic_keywords = ", ".join([self._feature_names[i] for i in top_indices if contextual_scores[i] > 0])
                
                results.append({
                    "Period": period,
                    "Category": cat,
                    "Topic_ID": t_idx + 1,
                    "Intensity_in_Slice": slice_topic_intensities[t_idx],
                    "Contextual_Keywords": topic_keywords
                })

        pd.DataFrame(results).to_csv(
            self.output_path + "lda_all_topics_keywords_by_period_category.csv", index=False
        )
        print("Comprehensive CSV saved with contextual keywords.")

    def visualization(self):
        """
        Refined visualization with balanced aspect ratio, 10 keywords, and optimized Y-axis
        """
        print("Generating visualization...")
        
        THEME_COLORS = ["#00798C", "#D1495B", "#30638E", "#EDAE49", "#66A182"]
        N_TOPICS = self.lda_model.n_components
        
        topic_cols = [f"Topic {i+1}" for i in range(N_TOPICS)]
        
        plot_df = pd.concat([self._df[["period", "categories"]], 
                            pd.DataFrame(self._topic_distributions, columns=topic_cols)], 
                           axis=1)
        plot_df = plot_df.groupby(["period", "categories"]).mean().reset_index()
        melted_df = plot_df.melt(id_vars=["period", "categories"], var_name="Topic", value_name="Intensity")

        sns.set_theme(style="whitegrid")
        
        g = sns.FacetGrid(
            melted_df, col="categories", hue="Topic", col_wrap=3, 
            height=4.5, aspect=1.2, sharey=True,
            palette=sns.color_palette(THEME_COLORS[:N_TOPICS])
        )
        
        g.map(sns.lineplot, "period", "Intensity", marker="o", linewidth=2.5)
        g.set(ylim=(0, None))

        g.fig.set_size_inches(16, 11) 
        g.fig.subplots_adjust(bottom=0.3, hspace=0.4, wspace=0.15) 

        # Top 10
        topic_labels = []
        for i, topic in enumerate(self.lda_model.components_):
            top_words = [self._feature_names[idx] for idx in topic.argsort()[:-11:-1]]
            topic_labels.append(f"Topic {i+1}: {', '.join(top_words)}")
        
        legend_text = "\n".join(topic_labels)

        g.fig.text(
            0.5, 0.04, 
            f"Topic Definition Reference (Top 10 Keywords):\n{legend_text}", 
            fontsize=10, linespacing=1.6, ha='center', va='bottom',
            bbox=dict(facecolor='whitesmoke', edgecolor='gray', alpha=0.9, boxstyle='round,pad=1')
        )

        g.add_legend(title="Topics", adjust_subtitles=True)
        g.set_axis_labels("Period", "Topic Intensity (Mean Distribution)")
        g.set_titles(col_template="{col_name}")

        save_path = os.path.join(self.output_path, "lda_category_period_evolution.png")
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
        self.save_output(n_top_words=10)
        self.visualization()
        print("=== LDA Pipeline Finished ===\n")