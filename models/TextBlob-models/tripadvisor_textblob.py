import pandas as pd
import time
from textblob import TextBlob
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the labeled dataset
dataset = pd.read_csv("processed_tripadvisor_dataset.csv")
#dataset = dataset.head(500)

# recording when model starts working
model_start_time = time.time()


# Defining function to predict sentiment
def predict_sentiment(text):
    # Creating TextBlob object
    blob = TextBlob(text)

    # Generating sentiment polarity from TextBlob object
    sentiment_polarity = blob.sentiment.polarity

    # Make prediction based on sentiment polarity calculations
    if sentiment_polarity <= (-0.6):
        return 1
    if sentiment_polarity >= (-0.6) and sentiment_polarity < (-0.2):
        return 2
    if sentiment_polarity >= (-0.2) and sentiment_polarity < 0.2:
        return 3
    if sentiment_polarity >= 0.2  and sentiment_polarity < 0.6:
        return 4
    if sentiment_polarity >= 0.6:
        return 5


    # Apply the predict_sentiment function to entire dataset
predicted_labels = dataset['review'].apply(predict_sentiment)

# Check if labels are correct to generate accuracy
accuracy = (predicted_labels == dataset['Rating']).mean()
predicted_labels = dataset['review'].apply(predict_sentiment)

# Calculate the precision, recall, and F1 score for each label
precision = []
recall = []
f1 = []
for i in range(1,6):
    actual_label = (dataset['Rating'] == i).astype(int)
    predicted_label = (predicted_labels == i).astype(int)
    precision.append(precision_score(actual_label, predicted_label))
    recall.append(recall_score(actual_label, predicted_label))
    f1.append(f1_score(actual_label, predicted_label))

# generating table of correct and incorrect predictions for each label
results = pd.crosstab(dataset['Rating'], predicted_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

print(results)

# print precision, recall, and F1 score for each label
print("Precision, ", precision)
print("Recall, ", recall)
print("F1 Score, ", f1)
print("accuracy, ", accuracy)

# recording when model finishes working
model_end_time = time.time()

print("time taken by model, " + str(model_end_time - model_start_time))
