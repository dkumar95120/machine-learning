#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features, sort_keys=True)
keys = sorted(data_dict.keys())
#print len(keys), len(data)

### your code below
mydata = []
for name, point in zip(keys, data):
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
 #   print name, salary, bonus
    mydata.append((name, salary, bonus))

from operator import itemgetter
sorted_data = sorted(mydata,key=itemgetter(2))
# print top 5 who got the highest bonus
print sorted_data[len(sorted_data)-1]
print sorted_data[len(sorted_data)-2]
print sorted_data[len(sorted_data)-3]
print sorted_data[len(sorted_data)-4]
print sorted_data[len(sorted_data)-5]
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()