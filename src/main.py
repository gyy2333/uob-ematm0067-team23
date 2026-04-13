import argparse
from pre_processing.data_understanding import main as load_data
from pre_processing.text_preprocessing import main as preprocessing
from modeling.modeling import Modeling
from visualization.plot_text import main as visualization


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--data-path",
    #     type=str,
    #     default="data/modeling/ai_ml_nlp_dataset_processed.csv",
    #     help="The path of the file of preprocessed data",
    # )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/",
        help="The directory saving output topic file and the figures",
    )
    args = parser.parse_args()

    # data_path = args.data_path
    # print("===input data path:", data_path)
    output_path = args.output_path
    print("===output data path:", output_path)

    load_data()
    preprocessing()

    bert_model = Modeling.create(
        "bert",
        data_path=output_path + "pre_processed/",
        output_path=output_path + "modeling/",
    )
    bert_model.process()

    lda_model = Modeling.create(
        "lda",
        data_path=output_path + "pre_processed/ai_ml_nlp_dataset_processed.csv",
        output_path=output_path + "modeling/",
    )
    lda_model.process()

    visualization(
        data_dir=output_path + "pre_processed/",
        plot_dir=output_path + "visualization/",
    )


if __name__ == "__main__":
    main()
