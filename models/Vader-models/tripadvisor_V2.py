import pandas as pd
import time
import matplotlib.pyplot as plt
# VADER imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score, f1_score

# read in dataset as pandas dataframe
dataset = pd.read_csv('processed_tripadvisor_dataset.csv')
print(dataset.shape)
#dataset = dataset.head(500)
print(dataset.shape)

# Declaring lists and variables required for accuracy, precision, recall and F-1 score calculations
predictions = []
true_labels = []
correctPredictions = [0]*5
totalPredictions = [0]*5
precision = [0]*5
recall = [0]*5
f1 = [0]*5

# defining function to generate sentiment scoring
def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # Generate sentiment polarity using above SentimentIntensityAnalyzer object
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # carry out predictions based on sentiment ratings
    if sentiment_dict['compound'] < (-0.6):
        predictions.append(1)

    elif sentiment_dict['compound'] >= (-0.6) and sentiment_dict['compound'] <= (-0.2) :
        predictions.append(2)

    elif sentiment_dict['compound'] >= (-0.2) and sentiment_dict['compound'] <= 0.2 :
        predictions.append(3)

    elif sentiment_dict['compound'] >= 0.2 and sentiment_dict['compound'] <= 0.6 :
        predictions.append(4)

    elif sentiment_dict['compound'] >= 0.6:
        predictions.append(5)

    true_labels.append(dataset['Rating'][i])

#recording when model starts working
model_start_time = time.time()

# Assigning predicted labels to reviews
for i in range(len(dataset)):
    #print("current data entry", + i)
    sentiment_scores(dataset['review'][i])

# Comparing and keeping track of correct predictions
for i in range(len(dataset)):
    if(true_labels[i] == predictions[i]):
        correctPredictions +=1;

for label in range(5):
    precision[label] = precision_score(true_labels, predictions, labels=[label+1], average='micro')
    recall[label] = recall_score(true_labels, predictions, labels=[label+1], average='micro')
    f1[label] = f1_score(true_labels, predictions, labels=[label+1], average='micro')

accuracy = correctPredictions / len(predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

for label in range(5):
    print("Label ", label+1)
    print("Precision:", precision[label])
    print("Recall:", recall[label])
    print("F1 score:", f1[label])
    print("Total number of predictions:", totalPredictions[label])
    print("Total number of correct predictions:", correctPredictions[label])


print("Total number correct of predictions by model", + correctPredictions)
results = pd.crosstab(pd.Series(true_labels), pd.Series(predictions), rownames=['Actual'], colnames=['Predicted'])

print(results)


#recording when model finishes working
model_end_time = time.time()

# Printing model runtime
print("time taken by model, " + str(model_end_time - model_start_time))

