#!/usr/bin/python

import sys
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from tester import test_classifier
sys.path.append("../tools/")
from sklearn.pipeline import make_pipeline, Pipeline
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments', 'total_stock_value', 'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop( "TOTAL", 0 )
### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict
keys = sorted(my_dataset.keys())
### add a feature for total_payments/salary to see who benefited the most radically due to non-salary compensation
fname = 'payment_to_salary'
for name in keys:
	if (my_dataset[name]['total_payments'] != 'NaN' and my_dataset[name]['salary'] != 'NaN'):
		my_dataset[name][fname] = float(my_dataset[name]['total_payments'])/my_dataset[name]['salary']
	else: # to avoid this person from being removed from the master list
		my_dataset[name][fname] = 0.000001
features_list.append(fname)
### add a feature for Total_stock/total_payments to see who had most to gain from the stock
fname = 'total_stock_to_payments'
for name in keys:
	if (my_dataset[name]['total_stock_value'] != 'NaN' and my_dataset[name]['total_payments'] != 'NaN'):
		my_dataset[name][fname] = float(my_dataset[name]['total_stock_value'])/my_dataset[name]['total_payments']
	else: # to avoid this person from being removed from the master list
		my_dataset[name][fname] = 0.000001
features_list.append(fname)

nfeat = len(features_list)
print "number of features:", nfeat
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# scale features to normalize them
#features = preprocessing.scale(np.array(features))
features = np.array(features)	
labels = np.array(labels)

nsample = len(labels)
print "number of keys:", len(keys)
print " number of samples:", nsample
print "emp name                        poi   payment  stock_value    poi_to_this  payment/salary total_stock/payments"
i = 0
for name in keys:
	print '{:3}'.format(i+1),
	print '{:20}'.format(name[:20]),
	print '{:>10}'.format(labels[i]),
	for k in range (0,nfeat-1):
		print '{:>10}'.format(features[i][k]),
	print ''
	i += 1

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_names = [
         	"Naive Bayes", 
			"SVM", 
         	"Decision Tree", 
			"Nearest Neighbors", 
         	"Random Forest", 
         	"AdaBoost",
         	"QDA"]
classifiers = [
	GaussianNB(),
    SVC (gamma=.1, C=.1, class_weight='balanced',random_state=42),
    DecisionTreeClassifier(min_samples_split=6, splitter='random', max_depth=2),
	KNeighborsClassifier(3, weights='distance'),
    RandomForestClassifier(n_estimators=10, min_samples_leaf=3, min_samples_split=3, random_state=42, class_weight='balanced'),
    AdaBoostClassifier(random_state=0),
    QuadraticDiscriminantAnalysis()]

# use a full grid over all parameters to tune classifier using GridSearchCV
# param_grid = {'random_state':[0,1]}

#minimum score required to consider classification a success
recall = precision = .3 

for do_pca in (0,1):
	for name, clf in zip(clf_names, classifiers):
		### Task 5: Tune your classifier to achieve better than .3 precision and recall 
		### using our testing script. Check the tester.py script in the final project
		### folder for details on the evaluation method, especially the test_classifier
		### function. Because of the small size of the dataset, the script uses
		### stratified shuffle split cross validation. For more info: 
		### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

		# Example starting point. Try investigating other evaluation techniques!

		print "Performing classification using", name,
		if (do_pca) : 
			print 'with pca'
		else:
			print ''
		features_train, features_test, labels_train, labels_test = \
		    train_test_split(features, labels, test_size=0.3, random_state=42)
		m_feat = nfeat-1
		###############################################################################
		if (do_pca):
			kbest = SelectKBest(k='all')
			pca = PCA(svd_solver='randomized',n_components=m_feat, whiten=True)
			steps=[('feature_selection',kbest),("pca", pca),('clf',clf)]
			clf = Pipeline(steps)

#		grid_search = GridSearchCV(clf, param_grid=None)
#		grid_search.fit(np.array(features_train), np.array(labels_train))

#		print "best score", grid_search.best_score_
#		print "best params", grid_search.best_params_

	### get classifier metrics from test_classifier
		clf_accuracy, clf_precision, clf_recall, clf_f1, clf_f2 = test_classifier (clf, my_dataset, features_list)

	### Task 6: Dump your classifier, dataset, and features_list so anyone can
	### check your results. You do not need to change anything below, but make sure
	### that the version of poi_id.py that you submit can be run on its own and
	### generates the necessary .pkl files for validating your results.
		if (clf_precision > precision and clf_recall > recall):
			print "***found better precision and recall:", clf_precision, clf_recall
			recall = clf_recall
			precision = clf_precision
			dump_classifier_and_data(clf, my_dataset, features_list)


