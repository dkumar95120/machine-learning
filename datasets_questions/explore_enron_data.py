#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print "number of people in enron data:", len(enron_data)
print "number of features in each person:", len(enron_data.items()[0][1])
print "features", enron_data.items()[0][1].keys()
npoi = 0
nsal = 0
nemail = 0
ntpay=0
nopoi=opoi=0
nrpoi=rpoi=0
nepoi=epoi=0
print "name                         poi    salary     payment  stock_value    poi_to_this"
for name in enron_data.keys():
	if (enron_data[name]["poi"] == 1): 
		npoi = npoi + 1
		if (enron_data[name]["other"] != "NaN"): opoi += 1
		if (enron_data[name]["shared_receipt_with_poi"] != "NaN"): rpoi += 1
		if (enron_data[name]["expenses"] != "NaN"): epoi += 1
		print name
		for feature in enron_data[name].keys():
			print '{0:25}:\t{1:>22}'.format(feature, enron_data[name][feature])
	else:
		if (enron_data[name]["other"] != "NaN"): nopoi += 1
		if (enron_data[name]["shared_receipt_with_poi"] != "NaN"): nrpoi += 1
		if (enron_data[name]["expenses"] != "NaN"): nepoi += 1
	if (enron_data[name]["salary"] != "NaN"): nsal = nsal + 1
	if (enron_data[name]["email_address"] != "NaN"): nemail = nemail + 1
	if (enron_data[name]["total_payments"] != "NaN"): ntpay = ntpay + 1

print "number of person of interest:", npoi
print "number of poi with other:", opoi
print "number of non-poi with other:", nopoi
print "number of poi with shared_receipt_with_poi:", rpoi
print "number of non-poi with shared_receipt_with_poi:", nrpoi
print "number of poi with expenses:", epoi
print "number of non-poi with expenses:", nepoi

poi_names={}
npoi = 0
npoi_found = 0
with open("..\\final_project\poi_names.txt", "r") as f:
	for line in f:
		if (line =="\n"): continue
		values = line.split()
		if (values[0][0] != '('): continue
		name = values[1] + values[2]
		poi_names[name] = values[0][1:2]
		npoi = npoi + 1
		if (values[0]=='(y)'): npoi_found = npoi_found + 1

print "number of person of interest:", npoi
print "number of person of interest found:", npoi_found
