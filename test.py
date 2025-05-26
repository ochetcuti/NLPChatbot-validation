import pandas as pd

df = pd.read_excel("dataset/SAD_v1_cleaned.xlsx")

# Checking corilation to ensure the normlaisation is applied
corr = df['avg_severity'].corr(df['avg_severity_normalised'])
# should be -1
print(f"Correlation between avg_severity and normalized: {corr:.2f}")