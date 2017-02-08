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
for name in enron_data.keys():
	if (enron_data[name]["poi"] == 1): npoi = npoi + 1
	if (enron_data[name]["salary"] != "NaN"): nsal = nsal + 1
	if (enron_data[name]["email_address"] != "NaN"): nemail = nemail + 1
	if ("JAMES" in name and "PRENTICE" in name): 
		print name, "stock value", enron_data[name]["total_stock_value"]
	if ("WESLEY" in name and "COLWELL" in name):
		print name, "messages to poi:", enron_data[name]["from_this_person_to_poi"]
	if ("SKILLING" in name and "JEFFREY" in name):
		print name, "Stock Options:", enron_data[name]["exercised_stock_options"]
		print name, "Total Payments:", enron_data[name]["total_payments"]
	if ("KENNETH" in name and "LAY" in name):
		print name, "Total Payments:", enron_data[name]["total_payments"]
	if ("FASTOW" in name):
		print name, "Total Payments:", enron_data[name]["total_payments"]

print "number of person of interest:", npoi
print "number of person getting salary:", nsal
print "number of person with email address:", nemail

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