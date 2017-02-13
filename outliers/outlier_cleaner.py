#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    print "in outlierCleaner"
    cleaned_data = []

    ### your code goes here
    for age, net_worth, prediction in zip(ages, net_worths, predictions):
        #print "Age:",age[0]
        #print "net worth:",net_worth[0]
        error = abs (net_worth[0] - prediction[0])
        #print "error:", error
        item = (age[0], net_worth[0], error)
        cleaned_data.append(item)
    
    from operator import itemgetter
    cleaned_data = sorted(cleaned_data,key=itemgetter(2))
    print "Sorted data:"
    for item in cleaned_data: 
        print item
    ntotal = len(cleaned_data)
    print "number of total points:", ntotal
    ntotal = .9*ntotal
    print "90 percent of total points:", ntotal
    cleaned_data = cleaned_data[:int(ntotal)]
    print "cleaned data:"
    for item in cleaned_data: 
        print item    

    return cleaned_data

