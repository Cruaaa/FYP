import pandas as pd
import time
import matplotlib.pyplot as plt
# VADER imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score

# read in dataset as pandas dataframe
dataset = pd.read_csv('processed_dataset.csv')
print(dataset.shape)
# dataset = dataset.head(500)
print(dataset.shape)

predictions = []
true_labels = []
correctPredictions = 0;


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # Generate sentiment polarity using above SentimentIntensityAnalyzer object
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # carry out predictions within sentiment boundaries
    if sentiment_dict['compound'] < (0.5):
        predictions.append('positive')

    else:
        predictions.append('negative')

    true_labels.append(dataset['sentiment'][i])


# recording when model starts working
model_start_time = time.time()

# Assigning predicted labels to reviews
for i in range(len(dataset)):
    # print("current data entry", + i)
    sentiment_scores(dataset['review'][i])

# Comparing and keeping track of correct predictions
for i in range(len(dataset)):
    if (true_labels[i] == predictions[i]):
        correctPredictions += 1;

# Calculate precision, recall, and F1 scores for positive predictions
precision_pos = precision_score(true_labels, predictions, pos_label='positive')
recall_pos = recall_score(true_labels, predictions, pos_label='positive')
f1_pos = f1_score(true_labels, predictions, pos_label='positive')

# Calculate precision, recall, and F1 scores for negative predictions
precision_neg = precision_score(true_labels, predictions, pos_label='negative')
recall_neg = recall_score(true_labels, predictions, pos_label='negative')
f1_neg = f1_score(true_labels, predictions, pos_label='negative')

accuracy = correctPredictions / len(predictions)

# Print the results
print("Positive Predictions")
print("Precision,", precision_pos)
print("Recall,", recall_pos)
print("F1 score,", f1_pos)

print("Negative Predictions")
print("Precision,", precision_neg)
print("Recall,", recall_neg)
print("F1 score,", f1_neg)

print("Total number correct of predictions by model", + correctPredictions)
results = pd.crosstab(pd.Series(true_labels), pd.Series(predictions), rownames=['Actual'], colnames=['Predicted'])

print(results)

# recording when model finishes working
model_end_time = time.time()

print("time taken by model, " + str(model_end_time - model_start_time))

