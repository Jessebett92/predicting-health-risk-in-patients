import pandas as pd

data = pd.read_csv("healthcare_real_time_dataset.csv")

print(data["Chronic Disease History"].value_counts())