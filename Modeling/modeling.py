import pandas as pd


class Modeling:
    _registry = {}  # 子类注册表

    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self._df = None
        self._model = None
        self._topics = None

    def __init_subclass__(cls, **kwargs):
        """自动注册子类"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__.lower()] = cls

    @classmethod
    def create(cls, model_type: str, **kwargs):
        """工厂方法：根据类型创建实例"""
        model_class = cls._registry.get(model_type.lower())
        if model_class is None:
            raise ValueError(
                f"未知类型: {model_type}，可用: {list(cls._registry.keys())}"
            )
        return model_class(**kwargs)

    def speak(self):
        raise NotImplementedError

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
