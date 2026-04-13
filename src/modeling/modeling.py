import pandas as pd
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CUSTOM_STOPWORDS = {
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
    "pronoun",
    # 8. Math & Formatting Fragments
    "math",
    "mathsf",
}
COMBINED_STOPWORDS = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))


class Modeling:
    _registry = {}  # Subclass registry

    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

        self._df = None
        self._model = None
        self._topics = None

    def __init_subclass__(cls, **kwargs):
        # Automatic subclass registration
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__.lower()] = cls

    @classmethod
    def create(cls, model_type: str, **kwargs):
        # Determine which subclass to create based on the input string.
        model_class = cls._registry.get(model_type.lower())
        if model_class is None:
            raise ValueError(
                f"未知类型: {model_type}，可用: {list(cls._registry.keys())}"
            )
        return model_class(**kwargs)

    def load_data(self):
        self._df = pd.read_csv(self.data_path)
        print("Dataset shape:", self._df.shape)
        print("Columns:", self._df.columns.tolist())

    def build_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def evaluation(self):
        raise NotImplementedError

    def visualization(self):
        raise NotImplementedError

    def save_output(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
