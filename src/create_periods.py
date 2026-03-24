import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/ai_ml_nlp_full_filtered.csv")

# Create period column
def get_period(year):

    if 2007 <= year <= 2011:
        return "2007_2011"
    elif 2012 <= year <= 2016:
        return "2012_2016"
    elif 2017 <= year <= 2020:
        return "2017_2020"
    else:
        return "2021_plus"


df["period"] = df["year"].apply(get_period)

# Show counts
print("Period counts:")
print(df["period"].value_counts())

# Save updated dataset
df.to_csv("data/processed/ai_ml_nlp_full_filtered.csv", index=False)

print("\nDataset saved with period column.")
print("New shape:", df.shape)
