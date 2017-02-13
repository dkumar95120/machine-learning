#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
import numpy
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys=True)
keys = sorted(data_dict.keys())
labels, features = targetFeatureSplit(data)

print "feature dimensions:", len(features)
print "labels dimensions:", len(labels)

features = numpy.reshape( numpy.array(features), (len(features), 1))
labels = numpy.reshape( numpy.array(labels), (len(labels), 1))
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!  
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

clf = DecisionTreeClassifier()
t0=time()	
clf = clf.fit(features_train, labels_train)
print "training time (SVC):", round(time()-t0, 3),"s"

t0 = time()
labels_pred = clf.predict(features_test)
print "prediction time:", round(time()-t0,3),"s"

print "real  pred"
labels_actual = []
labels_output = []
for real, pred in zip (labels_test, labels_pred):
	if real[0] > 0:
		labels_actual.append(real[0])
		labels_output.append(pred)
		print real[0], pred

score = accuracy_score(labels_output, labels_actual)
print "DT accuracy = ", score
print "number of predicted pois:", sum(labels_output), len(labels_output)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision, recall, fscore, support = precision_recall_fscore_support(labels_actual, labels_output, average='weighted')
print "precision:", precision
print "recall:", recall
print "fscore:", fscore