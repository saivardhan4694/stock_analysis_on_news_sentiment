import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the trained model and tokenizer
model_path = r"D:\repositories\stock_prediciton_on_news_sentiment\model_files"
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# 2. Load new data
new_data_path = r"D:\repositories\stock_prediciton_on_news_sentiment\Data\Test_data.csv"
new_data = pd.read_csv(new_data_path)
print(new_data.head())

# 3. Preprocess and tokenize the new data
texts = new_data['Title'].tolist()
true_labels = new_data['Sentiment'].tolist()
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='tf')

# 4. Make predictions
predictions = model.predict(encodings).logits
predicted_classes = np.argmax(predictions, axis=1)

# 5. Decode the predicted classes back to sentiment labels
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_labels = [label_map[pred] for pred in predicted_classes]

# Add the predictions to the DataFrame
new_data['Predicted Sentiment'] = predicted_labels

# 6. Calculate Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

# Generate the classification report
class_report = classification_report(true_labels, predicted_labels, zero_division=1)

# Create output folder
output_folder = r"model_test_reults"

# Save scores and classification report in a text file
metrics_text_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_text_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

print(f"Metrics and classification report saved to {metrics_text_path}")

# 7. Plot and Save Confusion Matrix
unique_labels = list(set(true_labels) | set(predicted_labels))
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=unique_labels)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
confusion_matrix_path = os.path.join(output_folder, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix saved to {confusion_matrix_path}")

# 8. Plot and Save Distribution of Predicted Sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='Predicted Sentiment', data=new_data, palette='viridis')
plt.title("Distribution of Predicted Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Count")
distribution_plot_path = os.path.join(output_folder, "class_distribution.png")
plt.savefig(distribution_plot_path)
plt.close()
print(f"Class distribution plot saved to {distribution_plot_path}")

# 9. Save the results to a new CSV file
output_csv_path = os.path.join(output_folder, "predicted_sentiments_with_metrics.csv")
new_data.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")
