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
nppoi=0
print "name                         poi    salary     payment  stock_value    poi_to_this"
for name in enron_data.keys():
	if (enron_data[name]['total_payments'] > 2000000 or enron_data[name]['from_poi_to_this_person'] > 50 or enron_data[name]['from_this_person_to_poi'] > 50):
		print '{0:20} {1:>10} {2:>10} {3:>10} {4:>10} {5:>10}'.format(name[:20], 
												enron_data[name]['poi'],
												enron_data[name]['salary'], 
												enron_data[name]['total_payments'], 
												enron_data[name]['total_stock_value'],
												enron_data[name]['from_poi_to_this_person'])
	if (enron_data[name]["poi"] == 1): 
		npoi = npoi + 1
		if (enron_data[name]["total_payments"] != "NaN"): nppoi = nppoi + 1
	if (enron_data[name]["salary"] != "NaN"): nsal = nsal + 1
	if (enron_data[name]["email_address"] != "NaN"): nemail = nemail + 1
	if (enron_data[name]["total_payments"] != "NaN"): ntpay = ntpay + 1

print "number of person of interest:", npoi
print "number of person getting salary:", nsal
print "number of person with email address:", nemail
print "number of person with total payments:", ntpay
print "number of poi with total payments:", nppoi

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
