import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("dataset/SAD_v1_cleaned.xlsx")

# all top lables in dataset
labels = df['top_label'].unique()
stress_counts = []
non_stress_counts = []

# Loop through each label
for label in labels:
    df_label = df[df['top_label'] == label]
    # split stressor to true and false
    counts = df_label['is_stressor'].value_counts()
    stress = counts.get(1, 0) #is stressor col
    non_stress = counts.get(0, 0)
    
    stress_counts.append(stress)
    non_stress_counts.append(non_stress)

data = pd.DataFrame({
    'Top Label': labels,
    'Stressors': stress_counts,
    'Non Stressors': non_stress_counts
})
data.set_index('Top Label')[['Non Stressors', 'Stressors']].plot(kind='bar', color=['#929591', "#4D4D4D"], stacked=True,figsize=(12, 8))
plt.title('Number of Stressors and Non Stressors by Top Label')
plt.xlabel('Top Label')
plt.ylabel('Number of datapoints')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reportImages/NumOfDatapointsByTopLabel.png")
plt.show()


# Stressors vs Non Stressors as a whole
total_stressors = data['Stressors'].sum()
total_non_stressors = data['Non Stressors'].sum()

labels = ['Stressor', 'Non Stressor']
sizes = [total_stressors, total_non_stressors]

plt.figure(figsize=(6, 6))
# 1dp percent
plt.pie(sizes,autopct='%1.1f%%',labels=labels,colors=['#929591', '#4D4D4D'])
plt.title('Number of Stressor vs Non Stressor Sentences')
plt.savefig("reportImages/NumOfStressorsVSnonStressors.png")
plt.show()


# Get the data for serverity of sentences
severity = df['avg_severity']
min_severity = severity.min()
max_severity = severity.max()
avg_severity = severity.mean()

print(f"Minimum severity (avg_severity): {min_severity}")
print(f"Maximum severity (avg_severity): {max_severity}")
print(f"Average severity (avg_severity): {avg_severity:.2f}")

plt.figure(figsize=(8, 5))
plt.hist(severity, bins=20, color='#929591')
plt.title('Distribution of Average Severity Scores')
plt.xlabel('Average Severity')
plt.ylabel('Number of Sentences')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("reportImages/distributionAvgSeverityScores.png")
plt.show()