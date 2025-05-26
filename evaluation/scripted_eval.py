import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, root_mean_squared_error, mean_absolute_error, make_scorer, mean_squared_error, r2_score
import pandas as pd
def evaluate_predictions(true_labels, predicted_labels, label_name, display_graph = True):
    # Is lable within the lower and upper bound (larger or equal to min and smaller than max)
    true_classification_labels = true_labels
    predicted_classification_labels = predicted_labels
    cm = confusion_matrix(true_classification_labels, predicted_classification_labels)
    accuracy = accuracy_score(true_classification_labels, predicted_classification_labels)
    error_rate = 1 - accuracy
    precision = precision_score(true_classification_labels, predicted_classification_labels)
    recall = recall_score(true_classification_labels, predicted_classification_labels)
    f1 = f1_score(true_classification_labels, predicted_classification_labels)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(label_name)
    print(f"Accuracy: {accuracy}\nError_rate: {error_rate}\nPrecision: {precision}\nRecall: {recall}\nSpecificity: {specificity}\nF1 Score {f1}")
    if display_graph:
        plt.figure(figsize=(8, 6))
        # Confusion Matrix
        conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        conf_matrix_display.plot(cmap='Greys', colorbar=False)
        plt.title(f"{label_name} Classification\nConfusion Matrix")
        plt.savefig(f"evaluation/plots/{label_name}_confusion_matrix.png")
        plt.close()
        # Performance Metrics Bar Chart

        plt.figure(figsize=(8, 6))
        metrics = ['Accuracy', 'Error Rate', 'Precision', 'Recall', 'Specificity', 'F1 Score']
        values = [accuracy, error_rate, precision, recall, specificity, f1]
        
        bars = plt.bar(metrics, values, color='#929591')
        plt.title(f"{label_name} Classification\nPerformance Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Annotate bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        plt.savefig(f"evaluation/plots/{label_name}_performance_metrics.png")
        plt.close() 
        
        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(true_labels, predicted_labels, alpha=0.6, color="#929591")
        plt.plot([-1, 1], [-1, 1], '--', color='gray')  # Diagonal line
        plt.xlabel('True Normalised Severity')
        plt.ylabel(f"{label_name} Compound Score")
        plt.title(f"{label_name} Regression\nCompound vs Actual Severity")
        plt.grid(True, linestyle='--')
        plt.savefig(f"evaluation/plots/{label_name}_scatter_regression.png")
        plt.close()
    return cm,accuracy,error_rate,precision,recall,specificity,f1
# save timing reports for each stress level
df=pd.read_csv('evaluation/scriptedTimingReport.csv')  
df['true_stress_binary'] = (df['true_stress'] == 'high_stress').astype(int)
df['predicted_stress_binary'] = (df['predicted_stress'] == 'high_stress').astype(int)
evaluate_predictions(df['true_stress_binary'], df['predicted_stress_binary'], label_name=f"Scripted Final Evaluation - High Stress")

df['true_stress_binary'] = (df['true_stress'] == 'moderate_stress').astype(int)
df['predicted_stress_binary'] = (df['predicted_stress'] == 'moderate_stress').astype(int)
evaluate_predictions(df['true_stress_binary'], df['predicted_stress_binary'], label_name=f"Scripted Final Evaluation - Moderate Stress")

df['true_stress_binary'] = (df['true_stress'] == 'low_stress').astype(int)
df['predicted_stress_binary'] = (df['predicted_stress'] == 'low_stress').astype(int)
evaluate_predictions(df['true_stress_binary'], df['predicted_stress_binary'], label_name=f"Scripted Final Evaluation - Low Stress")

average_time = df['response_time'].mean()
max_time = df['response_time'].max()

print(f"Average Time: {average_time:.4f} seconds")
print(f"Max Time: {max_time:.4f} seconds")