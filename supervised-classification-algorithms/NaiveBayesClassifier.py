#!/usr/bin/python

"""
   Use a Naive Bayes Classifier to identify emails by their authors
    authors and labels:
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


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

### create classifier
clf = GaussianNB()

## time to train the classifier
t0 = time()

clf.fit(features_train,labels_train)

print "training time : ", round(time()-t0,3), "s"

## time for prediction
t1 = time()

## predict function predicts the label for a particular feature
pred = clf.predict(features_test)

print "prediction time : ", round(time()-t1,3), "s"

### so we see in naive bayes algorithm prediction time takes less time than training time

# score will give accuracy
print clf.score(features_test,labels_test) ## which is 0.973265073948


matrix = confusion_matrix(labels_test,pred)
print matrix
print "Out of ", matrix[1][0]+matrix[1][1], " ", matrix[1][1] , " emails were predicted correctly and author is Sara "
print "Out of ", matrix[0][0]+matrix[0][1], " ", matrix[0][0] , " emails were predicted correctly and author is Chris "

report = classification_report(labels_test,pred)
print report
