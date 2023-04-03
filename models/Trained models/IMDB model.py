from sklearn import metrics
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.svm import LinearSVC

imdb_data = pd.read_csv('processed_dataset.csv')
x_train, x_test, y_train, y_test = train_test_split(imdb_data.review, imdb_data.sentiment, test_size=0.1)

# create the TfidfVectorizer object and fit it on the training data
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)

# transform the test data using the same fitted vectorizer object
x_test_tfidf = vectorizer.transform(x_test)

# convert sentiment labels to numerical values
y_train = (y_train.replace({'positive': 1, 'negative': 0})).values
y_test = (y_test.replace({'positive': 1, 'negative': 0})).values

#recording when model starts working
model_start_time = time.time()

# train a MultinomialNaiveBayes model
MNBmodel = MultinomialNB(alpha=10)
MNBmodel.fit(x_train_tfidf, y_train)

# evaluate the model on the test set
y_pred = MNBmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# check to ensure predictions and labels are being assigned correctly
#predicted_sentiment = MNBmodel.predict_proba(x_test_tfidf)[17]
#predicted_sentiment2 = MNBmodel.predict(x_test_tfidf)[17]
#print(predicted_sentiment)
#print(predicted_sentiment2)

# printing all relevant data to the model's predictions
print("STATISTICS FOR MNB MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Precision for negative class, {:.2f}%".format(precision[0] * 100))
print("Precision for positive class, {:.2f}%".format(precision[1] * 100))
print("Recall for negative class, {:.2f}%".format(recall[0] * 100))
print("Recall for positive class, {:.2f}%".format(recall[1] * 100))
print("F1 score for negative class, {:.2f}%".format(f1_score[0] * 100))
print("F1 score for positive class, {:.2f}%".format(f1_score[1] * 100))
print("Confusion matrix,")

# carrying out 10 fold cross validation
k_fold_acc_MNB = cross_val_score(MNBmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy, " + str(k_fold_acc_MNB))
print(confusion_matrix)

#recording when model finishes working
model_end_time = time.time()
# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))







#recording when model starts working
model_start_time = time.time()

# train a linear SupportVectorMachine model
SVMmodel = LinearSVC(max_iter=2000)
SVMmodel.fit(x_train_tfidf, y_train)

# evaluate the model on the test set
y_pred = SVMmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# printing all relevant data to the model's predictions
print("\n\n")
print("STATISTICS FOR SVM MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Precision for negative class, {:.2f}%".format(precision[0] * 100))
print("Precision for positive class, {:.2f}%".format(precision[1] * 100))
print("Recall for negative class, {:.2f}%".format(recall[0] * 100))
print("Recall for positive class, {:.2f}%".format(recall[1] * 100))
print("F1 score for negative class, {:.2f}%".format(f1_score[0] * 100))
print("F1 score for positive class, {:.2f}%".format(f1_score[1] * 100))
print("Confusion matrix,")

# carrying out 10 fold cross validation
k_fold_acc_SVM = cross_val_score(SVMmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy, " + str(k_fold_acc_SVM))
print(confusion_matrix)

#recording when model finishes working
model_end_time = time.time()

# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))






#recording when model starts working
model_start_time = time.time()

# train a linear SupportVectorMachine model
forestmodel = RandomForestClassifier(n_estimators=100, random_state=0)
forestmodel.fit(x_train_tfidf, y_train)

# evaluate the model on the test set
y_pred = forestmodel.predict(x_test_tfidf)

# generating accuracy, precision, recall and F-1 score
accuracy_score = metrics.accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# printing all relevant data to the model's predictions
print("\n\n")
print("STATISTICS FOR RANDOM FOREST MODEL")
print("Accuracy score, {:.2f}%".format(accuracy_score * 100))
print("Precision for negative class, {:.2f}%".format(precision[0] * 100))
print("Precision for positive class, {:.2f}%".format(precision[1] * 100))
print("Recall for negative class, {:.2f}%".format(recall[0] * 100))
print("Recall for positive class, {:.2f}%".format(recall[1] * 100))
print("F1 score for negative class, {:.2f}%".format(f1_score[0] * 100))
print("F1 score for positive class, {:.2f}%".format(f1_score[1] * 100))
print("Confusion matrix,")

# carrying out 10 fold cross validation
k_fold_acc_forest = cross_val_score(forestmodel, x_train_tfidf, y_train, cv=10)
print("kfold accuracy: " + str(k_fold_acc_forest))
print(confusion_matrix)

#recording when model finishes working
model_end_time = time.time()

# print how long the model took to run
print("time taken by model, " + str(model_end_time - model_start_time))