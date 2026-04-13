from .modeling import Modeling
from .modeling import COMBINED_STOPWORDS
import os
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


class BERT(Modeling):
    def __init__(self, data_path, output_path):
        super().__init__(data_path, output_path)
        print("Loading input data:", self.data_path)
        self.output_path = self.output_path + "bertopic/"
        os.makedirs(self.output_path, exist_ok=True)

        self._period_topics = None
        self._period_probs = None
        self._period_model = None

        self._global_topics = None
        self._global_probs = None
        self._global_model = None

        self._embedding_model = None
        self._glob_filt_txt = None
        self._period_filt_txt = None

    def build_model(
        self,
        model_name="all-MiniLM-L6-v2",
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        min_cluster_size=15,
    ):
        self._embedding_model = SentenceTransformer(model_name)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(stop_words=COMBINED_STOPWORDS)
        ctfidf_model = ClassTfidfTransformer()

        return BERTopic(
            embedding_model=self._embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            nr_topics="auto",
            verbose=False,
        )

    def train_model_period(
        self,
        period,
        text_col="clean_text",
        period_col="period",
    ):
        df_refined = self._df[self._df[period_col] == period].copy()
        self._period_filt_txt = df_refined[text_col].dropna().tolist()

        if len(self._period_filt_txt) < 20:
            print(
                f"Period {period}: only {len(self._period_filt_txt)} documents, skipping tarining BERT."
            )
            return False

        print(
            f"\n=== BERTopic analysis for period: {period} ({len(self._period_filt_txt)} docs) ==="
        )
        self._period_topics, self._period_probs = self._period_model.fit_transform(
            self._period_filt_txt
        )
        return True

    def train_model_global(
        self,
        text_col="clean_text",
        period_col="period",
    ):
        if period_col not in self._df.columns:
            raise ValueError(f"Column '{period_col}' not found in dataframe.")

        self._glob_filt_txt = self._df[text_col].fillna("").tolist()
        print(
            f"Fitting global BERTopic model to {len(self._glob_filt_txt)} documents..."
        )

        self._global_topics, self._global_probs = self._global_model.fit_transform(
            self._glob_filt_txt
        )

    def visualization(self, model, texts, period=None):
        save_fig_path = self.output_path + "figures/" + period + "/"
        os.makedirs(save_fig_path, exist_ok=True)
        bar_chart = model.visualize_barchart()
        bar_chart.write_html(save_fig_path + "bar_chart.html")
        # embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        # topic_cluster = model.visualize_documents(texts, embeddings=embeddings)
        # topic_cluster.write_html(save_fig_path + "topic_cluster.html")
        # topic_cluster.write_image(save_fig_path + "topic_cluster.png")
        hierarchy = model.visualize_hierarchy()
        hierarchy.write_html(save_fig_path + "hierarchy.html")
        heat_map = model.visualize_heatmap()
        heat_map.write_html(save_fig_path + "heat_map.html")

    def save_output(self, model, period):
        topic_info = model.get_topic_info()
        topic_info.to_csv(
            self.output_path + f"Modelingbertopic_{period}_topic_info.csv",
            index=False,
        )

        summary = []
        for _, row in topic_info.iterrows():
            if row.Topic == -1:
                continue
            summary.append(
                {
                    "period": period,
                    "topic": int(row.Topic),
                    "count": int(row.Count),
                    "name": row.Name,
                }
            )
        summary_df = pd.DataFrame(summary)
        summary_df_path = self.output_path + f"Modelingbertopic_topics_by_{period}.csv"
        summary_df.to_csv(summary_df_path, index=False)

    def process(self):
        self.load_data()

        periods = sorted(self._df["period"].dropna().unique())
        for period in periods:
            self._period_model = self.build_model(n_neighbors=20, min_cluster_size=20)
            train_success = self.train_model_period(period)
            if not train_success:
                print(f"[{period}]: no saving output files and figures")
                continue
            self.save_output(self._period_model, period)
            self.visualization(self._period_model, self._period_filt_txt, period)
            print(f"[{period}]: finish saving output files and figures")

        self._global_model = self.build_model(n_neighbors=40, min_cluster_size=40)
        self.train_model_global()
        self.visualization(self._global_model, self._glob_filt_txt, "global")
        self.save_output(self._global_model, "global")
