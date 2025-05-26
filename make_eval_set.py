import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("dataset/SAD_v1_cleaned.xlsx")

# Create Databin with a 3 quantile split (Low, average, high)
# https://pandas.pydata.org/docs/reference/api/pandas.qcut.html
df['severity_bin'] = pd.qcut(df['avg_severity_normalised'], q=3, labels=False)

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['severity_bin'], random_state=42)

# Ensure a even distribution
print(f"Train avg_severity mean: {train_df['avg_severity_normalised'].mean():.2f}")
print(f"Test avg_severity mean: {test_df['avg_severity_normalised'].mean():.2f}")

# Ensure seeding is not skewed
print(f"Train seed to non seed ratio: {train_df['is_seed'].value_counts(normalize=True)}")
print(f"Test seed to non seed ratio: {test_df['is_seed'].value_counts(normalize=True)}")

# Remove once split
train_df = train_df.drop(columns=['severity_bin'])
test_df = test_df.drop(columns=['severity_bin'])

train_df.to_excel("dataset/SAD_semantic_dataset.xlsx", index=False)
test_df.to_excel("dataset/SAD_semantic_dataset_test.xlsx", index=False)