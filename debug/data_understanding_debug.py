"""
Remove absolute path restrictions to ensure the code runs properly on any computer
To improve speed, download the database locally and modify the path accordingly.
Link to https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021
"""

import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# LOAD DATA


def load_data(path):
    df = pd.read_json(path, lines=True)
    print("Dataset Shape:", df.shape)
    print(df.info())
    print("Looking into the columns:", df.columns)
    print("number of missing values in each row are : ", df.isnull().sum())
    return df


## as we can see there are many missing values
## but the coulums we need don't have any missing values
## so we will remove the coloums that is not required for the analysis and keep only the one that is needed.
##  SELECTED REQUIRED XOLUMNS ARE - id, abstract, categories.


def select_columns(df):
    columns_to_keep = ["id", "abstract", "categories"]
    return df[columns_to_keep].copy()


## as one of the part is splitting corpus into periods - we need to extract the year  from id
## as there 2 different format for the id


# EXTRACT YEAR
def extract_year_safe(arxiv_id):
    arxiv_id = str(arxiv_id).split(":")[-1]

    if "/" in arxiv_id:
        year_str = arxiv_id.split("/")[-1][:2]
    else:
        year_str = arxiv_id[:2]
    try:
        y = int(year_str)
        return 1900 + y if y > 90 else 2000 + y

    except:
        return None


def add_year(df):
    df["year"] = df["id"].apply(extract_year_safe)
    print("Min year:", df["year"].min())
    print("Max year:", df["year"].max())
    return df


## as the age gap is 30 years we will be dividing it into 7 to 8 years max
# CREATING TIME PERIODS
def get_period(year):
    if 1991 <= year <= 1998:
        return "1991_1998"
    elif 1999 <= year <= 2006:
        return "1999_2006"
    elif 2007 <= year <= 2014:
        return "2007_2014"
    elif 2015 <= year <= 2021:
        return "2015_2021"
    else:
        return "unknown"


def add_period(df):
    df["period"] = df["year"].apply(get_period)
    return df


## analysing based ok the year extraction
## plot to find out the range graph for the count of paper in the per year
# VISUALISATION


def plot_year_distribution(df):
    year_counts = df["year"].value_counts().sort_index()
    plt.figure()
    plt.plot(year_counts.index, year_counts.values, marker="o")
    plt.title("Number of Papers per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.show()


##Papers Per Period


def plot_period_distribution(df):
    period_counts = df["period"].value_counts().sort_index()
    plt.figure()
    period_counts.plot(kind="bar")
    plt.title("Number of Papers per Time Period")
    plt.xlabel("Period")
    plt.ylabel("Count")
    plt.show()


##Abstract Length Over Time


def plot_period_distribution(df):
    df["abstract_length"] = df["abstract"].apply(len)
    plt.figure()
    df.groupby("year")["abstract_length"].mean().plot()
    plt.title("Average Abstract Length Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Length")
    plt.show()


## abstract length by time period
def plot_abstract_length(df):
    plt.figure()
    df.boxplot(column="abstract_length", by="period")
    plt.title("Abstract Length Distribution by Period")
    plt.suptitle("")
    plt.xlabel("Period")
    plt.ylabel("Length")
    plt.show()


## listing out count of all the unique categories from the category column.
# CATEGORY CLEANING


def clean_categories(cat):
    result = []

    if isinstance(cat, list):
        for item in cat:
            if isinstance(item, str):
                result.extend(item.split())

    elif isinstance(cat, str):
        result.extend(cat.split())

    return result


def process_categories(df):
    df["categories"] = df["categories"].apply(clean_categories)

    exploded_df = df.explode("categories").reset_index(drop=True)

    print("Unique categories:", len(exploded_df["categories"].unique()))

    return exploded_df


## now based on the subcategory in cs i am going selected the the paper which is related to AI, ML and NLP
# FILTER TARGET CATEGORY
def filter_ai_ml_nlp(df):
    TARGET_SUBCATS = ["cs.AI", "cs.LG", "cs.CL"]

    final_df = df[df["categories"].isin(TARGET_SUBCATS)]

    print("Category counts : \n", final_df["categories"].value_counts())
    print("Shape before duplicates:", final_df.shape)

    ## to make sure we don't have duplicate paper
    final_df = final_df.drop_duplicates(subset=["id"])
    print("Shape after duplicates", final_df.shape)
    return final_df


## data balancer checker - to reduce bais
# DATA IMBLANCE CHECK
def check_imbalance(final_df):
    print(final_df["period"].value_counts())
    print(final_df["period"].value_counts(normalize=True))

    ## as we can see the data is not balanced over time we r trying to balance it
    trend = pd.crosstab(final_df["period"], final_df["categories"])
    trend_norm = trend.div(trend.sum(axis=1), axis=0)
    trend_norm.plot(kind="bar", stacked=True)
    plt.title("Normalized chategory Distribution")
    plt.show()
    return trend_norm


## saving the final data to csv file
# SAVING THE DATA IN .CSV FILE
def save_data(df):
    os.makedirs("../data/raw_data/", exist_ok=True)
    df.to_csv("../data/raw_data/ai_ml_nlp_dataset.csv", index=False)
    print("Dataset saved sucessfully!")


## MAIN FUNCTION


def main():

    path = "hf://datasets/gfissore/arxiv-abstracts-2021/arxiv-abstracts.jsonl.gz"

    df = load_data(path)
    df = select_columns(df)
    df = add_year(df)
    df = add_period(df)

    plot_year_distribution(df)
    plot_period_distribution(df)
    plot_abstract_length(df)

    df = process_categories(df)
    final_df = filter_ai_ml_nlp(df)

    check_imbalance(final_df)

    save_data(final_df)


if __name__ == "__main__":
    main()
