#!/usr/bin/python

"""
    Use a SVM to identify emails from the Enron corpus by their authors:

    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# clf = svm.SVC(kernel="linear")
clf = svm.SVC(kernel="rbf", C=10000)

trainingTime = time()
clf.fit(features_train,labels_train)
print "training time : ", round(time()-trainingTime,3), "s"

predictionTime = time()
pred = clf.predict(features_test)
print "prediction time : ", round(time()-predictionTime,3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,labels_test)
print "accuracy of svm is : ", acc   ## 0.990898748578

# SVM is MUCH slower than Naive Bayes to train and use for predicting.


matrix = confusion_matrix(labels_test,pred)
print matrix
print "Out of ", matrix[1][0]+matrix[1][1], " ", matrix[1][1] , " emails were predicted correctly and author is Sara "
print "Out of ", matrix[0][0]+matrix[0][1], " ", matrix[0][0] , " emails were predicted correctly and author is Chris "

report = classification_report(labels_test,pred)
print report
