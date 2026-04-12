import argparse
from Modeling.modeling import Modeling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/modeling/ai_ml_nlp_dataset_processed.csv",
        help="The path of the file of preprocessed data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/",
        help="The directory saving output topic file and the figures",
    )
    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path

    print("===input data path:", data_path)
    print("===output data path:", output_path)

    bert_model = Modeling.create("bert", data_path=data_path, output_path=output_path)
    bert_model.process()

    lda_model = Modeling.create("lda", data_path=data_path, output_path=output_path)
    lda_model.process()


if __name__ == "__main__":
    main()
