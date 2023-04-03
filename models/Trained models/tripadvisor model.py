import time
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# reading in the data as a pandas dataframe
imdb_data = pd.read_csv('processed_tripadvisor_dataset.csv')

# line to reduce the dataframe size for testing
#imdb_data = imdb_data.head(500)
#print(imdb_data.shape)

# splitting data into training and testing data, 10% holdout for testing
x_train, x_test, y_train, y_test = train_test_split(imdb_data['review'], imdb_data['Rating'], test_size=0.1)


# creating the TfidfVectorizer object and fitting it on the training data
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)

# transforming the test data using the same fitted vectorizer object
x_test_tfidf = vectorizer.transform(x_test)



#recording when model starts working
model_start_time = time.time()

# training a MultinomialNaiveBayes model
MNBmodel = MultinomialNB(alpha = 0.01)
MNBmodel.fit(x_train_tfidf, y_train)

# evaluating the model on the test set
y_pred = MNBmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)

# Generating confusion matrix for predictions
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# check to ensure predictions and labels are being assigned correctly
#predicted_sentiment = MNBmodel.predict_proba(x_test_tfidf)[17]
#predicted_sentiment2 = MNBmodel.predict(x_test_tfidf)[17]
#print(predicted_sentiment)
#print(predicted_sentiment2)

# printing all relevant data to the model's predictions
print("STATISTICS FOR MNB MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Average precision for model, {:.2f}%".format(precision * 100))
print("Average recall for model, {:.2f}%".format(recall * 100))
print("Average F1 score for model, {:.2f}%".format(f1_score * 100))
print("Confusion matrix,")
print(confusion_matrix)

# carrying out 10 fold cross validation
k_fold_acc_MNB = cross_val_score(MNBmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy, " + str(k_fold_acc_MNB))
print(confusion_matrix)

#recording when model finishes working
model_end_time = time.time()

# Generating classification report of precision, recall and F-1 values for each label
classification_report_MNB = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_MNB)

# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))







#recording when model starts working
model_start_time = time.time()

# train a linear SupportVectorMachine model
SVMmodel = LinearSVC(random_state=0)
SVMmodel.fit(x_train_tfidf, y_train)

# evaluate the model on the test set
y_pred = SVMmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)

# Generating confusion matrix for predictions
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

#print(y_test.shape, x_test.shape)
#print(y_train.shape, x_train.shape)

# printing all relevant data to the model's predictions
print("\n\n")
print("STATISTICS FOR SVM MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Average precision for model, {:.2f}%".format(precision * 100))
print("Average recall for model, {:.2f}%".format(recall * 100))
print("Average F1 score for model, {:.2f}%".format(f1_score * 100))
print("Confusion matrix,")

# carrying out 10 fold cross validation
k_fold_acc_SVM = cross_val_score(SVMmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy, " + str(k_fold_acc_SVM))
print(confusion_matrix)

# Generating classification report of precision, recall and F-1 values for each label
classification_report_SVM = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_SVM)

#recording when model finishes working
model_end_time = time.time()

# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))






#recording when model starts working
model_start_time = time.time()

# train a linear Random Forest Classifier model
forestmodel = RandomForestClassifier(n_estimators=100, random_state=0)
forestmodel.fit(x_train_tfidf, y_train)
# evaluate the model on the test set
y_pred = forestmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=1)

# Generating confusion matrix for predictions
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# printing all relevant data to the model's predictions
print("\n\n")
print("STATISTICS FOR RANDOM FOREST MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Average precision for model, {:.2f}%".format(precision * 100))
print("Average recall for model, {:.2f}%".format(recall * 100))
print("Average F1 score for model, {:.2f}%".format(f1_score * 100))
print("Confusion matrix,")

# carrying out 10 fold cross validation
k_fold_acc_forest = cross_val_score(forestmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy: " + str(k_fold_acc_forest))
print(confusion_matrix)

# Generating classification report of precision, recall and F-1 values for each label
classification_report_forest = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_forest)

#recording when model finishes working
model_end_time = time.time()

# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))

