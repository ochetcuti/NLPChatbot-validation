import pandas as pd
df = pd.read_excel("dataset/SAD_v1_cleaned.xlsx")

# normalised_value = ((min - value) / (max - min)) * 2 - 1 | -1 to 1
# X prime is the set of all values you get by applying the function f to each element x in the set X.
min_sev = df['avg_severity'].min()
max_sev = df['avg_severity'].max()

df['avg_severity_normalised'] = ((max_sev - df['avg_severity']) / (max_sev - min_sev)) * 2 - 1

print(df[['avg_severity', 'avg_severity_normalised']].head())
print("New min:", df['avg_severity_normalised'].min())
print("New max:", df['avg_severity_normalised'].max())

df.to_excel("dataset/SAD_v1_cleaned.xlsx", index=False)