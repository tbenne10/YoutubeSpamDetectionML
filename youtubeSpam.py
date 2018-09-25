import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("directory-to-spam-csv")
#print (data.head())
X = data["CONTENT"]
Y = data["CLASS"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 5)
Y_train = Y_train.astype('int')
##Count Vectorization on training sample
cVec = CountVectorizer()
cVec
X_cVec = cVec.fit_transform(X_train)
cAr = X_cVec.toarray()


##TFID vectorization on training sample
tVec = TfidfVectorizer()
X_tVec = tVec.fit_transform(X_train)
tAr = X_tVec.toarray()

##TFID Vectorization with tuned hyperparameters
tttVec = TfidfVectorizer()
X_tttVec = tttVec.fit_transform(X_train)
tttAr = X_tttVec.toarray()

#smooth_idf = False, max_df = .1, sublinear_tf = True, stop_words = 'english', lowercase = False


########################
#Code for testing sample:
########################

# **Multinomial NB**
#m = MultinomialNB()
#m.fit(tttAr, Y_train)
#Y_pred = m.predict(X_tttVec)
#a = accuracy_score(Y_train, Y_pred)

## **SVM** 
s = LinearSVC()
s.fit(tttAr, Y_train)
Y_pred = s.predict(X_tttVec)
a = accuracy_score(Y_train, Y_pred)

#graphing
plt.scatter(Y_pred, a)
plt.title("Neural Network")
plt.xlabel('Attribute Index')
plt.ylabel('Regression Coefficients')
plt.show() 


## **F1 scoring**
f1 = f1_score(Y_train, Y_pred)

## **Confusion matrix
C = confusion_matrix(Y_train, Y_pred)
prediction = C[1][1] / (C[0][1] + C[1][1])
recall = C[1][1] / (C[1][0] + C[1][1])
alarmRate = C[0][1] / (C[0][1] + C[1][1])

print("f1:")
print(f1)
print("prediction:")
print(prediction)
print("recall:")
print(recall)
print("alarm rate:")
print(alarmRate)