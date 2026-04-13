from .modeling import Modeling


class LDA(Modeling):

    def __init__(self, data_path, output_path):
        super().__init__(data_path, output_path)
        self.output_path = self.output_path + "lda/"

    def process(self):
        print("Here is LDA")
