#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list, sort_keys=True )
poi, finance_features = targetFeatureSplit( data )
keys = sorted(data_dict.keys())
from sklearn.preprocessing import MinMaxScaler
fin_feat = numpy.array(finance_features)
xnorm = MinMaxScaler().fit(fin_feat)
fin_feat =  xnorm.transform(fin_feat)
X = numpy.array([200000., 1000000.])
X = xnorm.transform(X)
print X
nemp = len(keys)
print "Number of employees:", nemp, "compensated:", len(finance_features)
### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
esop = []
for key, fval in zip(keys, fin_feat):
    f1 = fval[0]  # salary
    f2 = fval[1]  # excercised stock options
#    f3 = fval[2]  # total_payments
    plt.scatter( f1, f2 )
    esop.append((key,f1))
plt.show()

from operator import itemgetter
sorted_emp = sorted(esop,key=itemgetter(1))

# print top 5 who excercised the highest number stock options
for item in sorted_emp:
    print item

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(fin_feat)
pred = kmeans.predict(fin_feat)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, fin_feat, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
