import pandas as pd
import re

# Load the dataset
file_path = "dataset/SAD_v1.xlsx"
df = pd.read_excel(file_path)

# Remove COVID
df_clean = df[df['is_covid'] != 1]
# remove columns not required
df_clean = df_clean.drop(columns=['is_covid', 'is_covid_conf', 'sID'])

# KEEP
keep_categories = {
    "Work",
    "Health, Fatigue, or Physical Pain",
    "Emotional Turmoil",
    "School"
}

# filter function on keep cats
def should_keep(row):
    labels = {str(row['top_label']), str(row['second_label'])}
    if (labels and keep_categories):
        return True
    return False

# for each item apply filter
# index must be reset due to lines being removed
df_clean = df_clean[df_clean.apply(should_keep, axis=1)].reset_index(drop=True)

# Remove na or empty fields
def is_valid(row):
    if (pd.isna(row['sentence']) or row['sentence'] == " " or row['sentence'] == ""):
        return False
    # no sentence 3 words or less
    elif (len(row['sentence'].strip().split()) <= 3):
        return False
    return True

df_clean = df_clean[df_clean.apply(is_valid, axis=1)].reset_index(drop=True)

# Remove rows with high SD (noisy data) where scorers have disagreed when rating data
df_clean = df_clean[df_clean['SD_severity'] <= 3].reset_index(drop=True)

# Save cleaned data for eval and model use
df_clean.to_excel("dataset/SAD_v1_cleaned.xlsx", index=False)

print(f"Final cleaned dataset rows: {len(df_clean)}")
df_clean.head()
