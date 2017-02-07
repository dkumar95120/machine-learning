#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

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
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###
from sklearn import svm

#crange = [10.,100.,1000.,10000.]
#for c in crange:
clf = svm.SVC(kernel='rbf',C=10000.)
t0=time()	
clf.fit(features_train,labels_train)
print "training time (SVC):", round(time()-t0, 3),"s"
#########################################################
t0 = time()
labels_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0,3),"s"
from sklearn.metrics import accuracy_score 
score = accuracy_score(labels_test, labels_pred)
print "C=10000, accuracy = ", score
authors = ["Sara","Chris"]
nc = 0
for pred in labels_pred:
	if (pred == 1): nc = nc + 1
print "Numer of Chris emails:", nc
print "10th email written by:", authors[labels_pred[10]]
print "26th email written by:", authors[labels_pred[26]]
print "50th email written by:", authors[labels_pred[50]]