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

        self._topics = None
        self._probs = None
        self._model = None

        self._global_topics = None
        self._global_probs = None
        self._global_model = None

        self._embedding_model = None
        self._glob_filt_txt = None
        self._filt_raw_txt = None
        self._filt_clean_txt = None
        self._merged_topic_rows = []

    def build_model(
        self,
        model_name="BAAI/bge-base-en-v1.5",  # "all-MiniLM-L6-v2",
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
            cluster_selection_method="eom",  # "leaf"
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(
            stop_words=COMBINED_STOPWORDS, ngram_range=(1, 2)
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        return BERTopic(
            embedding_model=self._embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            nr_topics=None,  # auto
            verbose=False,
        )

    def extract_train_data(self, period, category):
        obs_refined = self._df.loc[
            (self._df["period"] == period) & (self._df["categories"] == category)
        ].dropna(subset=["abstract", "clean_text"])
        self._filt_raw_txt = obs_refined["abstract"].tolist()
        self._filt_clean_txt = obs_refined["clean_text"].tolist()

    def train_model(self, period, category):
        if len(self._filt_raw_txt) < 20:
            print(
                f"Period {period} Category {category}: only {len(self._filt_raw_txt)} documents, skipping tarining BERT."
            )
            return False

        print(
            f"\n=== BERTopic analysis for period: {period} Category {category} ({len(self._filt_raw_txt)} docs) ==="
        )
        embeddings = self._embedding_model.encode(
            self._filt_raw_txt, show_progress_bar=True
        )
        self._topics, self._probs = self._model.fit_transform(
            self._filt_clean_txt, embeddings
        )
        n_valid = sum(1 for t in self._topics if t != -1)
        if n_valid == 0:
            print(
                f"Period {period} Category {category}: all documents are outliers (Topic=-1), skipping."
            )
            return False
        return True

    def train_global_model(
        self,
        obs_name="clean_text",
        variable_name="period",
    ):
        if variable_name not in self._df.columns:
            raise ValueError(f"Column '{variable_name}' not found in dataframe.")

        self._glob_filt_txt = self._df[obs_name].fillna("").tolist()
        print(
            f"Fitting global BERTopic model to {len(self._glob_filt_txt)} documents..."
        )

        self._global_topics, self._global_probs = self._global_model.fit_transform(
            self._glob_filt_txt
        )

    def visualization(self, model, texts, variable=None):
        save_fig_path = self.output_path + "figures/"
        os.makedirs(save_fig_path, exist_ok=True)
        bar_chart = model.visualize_barchart()
        bar_chart.write_html(save_fig_path + f"{variable}_bar_chart.html")
        # embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        # topic_cluster = model.visualize_documents(texts, embeddings=embeddings)
        # topic_cluster.write_html(save_fig_path + "topic_cluster.html")
        # topic_cluster.write_image(save_fig_path + "topic_cluster.png")
        hierarchy = model.visualize_hierarchy()
        hierarchy.write_html(save_fig_path + f"{variable}_hierarchy.html")
        heat_map = model.visualize_heatmap()
        heat_map.write_html(save_fig_path + f"{variable}_heat_map.html")

    def save_output(self, model, variable):
        topic_info = model.get_topic_info()
        topic_info["Total"] = topic_info["Count"].sum()
        topic_info.to_csv(
            self.output_path + f"Modelingbertopic_{variable}_topic_info.csv",
            index=False,
        )

        summary = []
        for _, row in topic_info.iterrows():
            if row.Topic == -1:
                continue
            summary.append(
                {
                    "period_category": variable,
                    "topic": int(row.Topic),
                    "count": int(row.Count),
                    "name": row.Name,
                }
            )
        summary_df = pd.DataFrame(summary)
        summary_df_path = (
            self.output_path + f"Modelingbertopic_topics_by_{variable}.csv"
        )
        summary_df.to_csv(summary_df_path, index=False)

        self.generate_merged_topics(topic_info, variable)
        print(f"finish saving {variable} output files and figures")

    def generate_merged_topics(self, topic_info, variable):
        parts = variable.split("_", 2)
        if len(parts) < 3:
            return

        period = f"{parts[0]}_{parts[1]}"
        category = parts[2]
        total_num = int(topic_info["Count"].sum())
        if total_num <= 0:
            return

        for _, row in topic_info.iterrows():
            topic_id = int(row.Topic)
            if topic_id < 0 or topic_id > 4:
                continue

            keywords = str(row.get("Representation"))
            keywords = (
                keywords.replace("'", "")
                .replace('"', "")
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
            )

            self._merged_topic_rows.append(
                {
                    "Period": period,
                    "Category": category,
                    "Topic_ID": topic_id + 1,
                    "Intensity_in_Slice": int(row.Count) / total_num,
                    "Contextual_Keywords": keywords,
                    "Name": row.Name,
                    "Count": int(row.Count),
                    "Total": total_num,
                }
            )
        if not self._merged_topic_rows:
            return

        merged_df = pd.DataFrame(self._merged_topic_rows)
        merged_df = merged_df.sort_values(["Period", "Category", "Topic_ID"])
        merged_path = self.output_path + "Modelingbertopic_topics_merged.csv"
        merged_df.to_csv(merged_path, index=False)

    def process(self):
        self.load_data()
        self._model = self.build_model(n_neighbors=20, min_cluster_size=20)

        periods = sorted(self._df["period"].dropna().unique())
        categories = self._df["categories"].dropna().unique()
        for period in periods:
            for category in categories:
                self.extract_train_data(period, category)
                train_success = self.train_model(period, category)
                output_file_name = period + "_" + category
                if not train_success:
                    print(f"no saving {output_file_name} output files and figures")
                    continue
                self.save_output(self._model, output_file_name)
                # self.visualization(self._model, self._filt_clean_txt, output_file_name)
