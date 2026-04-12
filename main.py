from Modeling.modeling import Modeling


def main():
    data_path = "data/modeling/ai_ml_nlp_dataset_processed.csv"
    output_path = "outputs/"
    model1 = Modeling.create("lda", data_path=data_path, output_path=output_path)
    print(model1.speak())

    model2 = Modeling.create("bert", data_path=data_path, output_path=output_path)
    print(model2.speak())
    model2.process()


if __name__ == "__main__":
    main()
