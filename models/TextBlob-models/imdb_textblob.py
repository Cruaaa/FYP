import pandas as pd
import time
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the labeled dataset
dataset = pd.read_csv("processed_dataset.csv")
dataset = dataset.head(500)

# recording when model starts working
model_start_time = time.time()


# Defining function to predict sentiment
def predict_sentiment(text):
    # Creating TextBlob object
    blob = TextBlob(text)
    # Generating sentiment polarity from TextBlob object
    sentiment_polarity = blob.sentiment.polarity

    # Make prediction based on sentiment polarity calculations
    if sentiment_polarity >= 0.0:
        return 'positive'
    else:
        return 'negative'

    # Apply the predict_sentiment function to entire dataset
predicted_labels = dataset['review'].apply(predict_sentiment)

# Check if labels are correct to generate accuracy
accuracy = (predicted_labels == dataset['sentiment']).mean()
predicted_labels = dataset['review'].apply(predict_sentiment)

# Calculate the precision, recall, and F1 score
pos_precision = precision_score(dataset['sentiment'], predicted_labels, pos_label='positive')
neg_precision = precision_score(dataset['sentiment'], predicted_labels, pos_label='negative')
pos_recall = recall_score(dataset['sentiment'], predicted_labels, pos_label='positive')
neg_recall = recall_score(dataset['sentiment'], predicted_labels, pos_label='negative')
pos_f1 = f1_score(dataset['sentiment'], predicted_labels, pos_label='positive')
neg_f1 = f1_score(dataset['sentiment'], predicted_labels, pos_label='negative')


# generating table of correct and incorrect predictions for each label
results = pd.crosstab(dataset['sentiment'], predicted_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

print(results)

# Print metrics of model
print(f"Positive precision, {pos_precision:.2f}")
print(f"Negative precision, {neg_precision:.2f}")
print(f"Positive recall, {pos_recall:.2f}")
print(f"Negative recall, {neg_recall:.2f}")
print(f"Positive F1 score, {pos_f1:.2f}")
print(f"Negative F1 score, {neg_f1:.2f}")
print(f"accuracy, {accuracy:.2f}")

# recording when model finishes working
model_end_time = time.time()

# Print time taken by model
print("time taken by model, " + str(model_end_time - model_start_time))
