<p><strong>Introduction to Machine Learning @Udacity</strong></p>
<p><strong>Final Project: Identify suspects in Enron Fraud</strong></p>
<p><strong>1. Dataset and goal of project</strong></p>
<p><strong>Goal</strong></p>
<p>The main purpose of project is develop the machine learning algorithm to detect person of interest(POI) from dataset. A POI is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.</p>
<p><strong>Dataset</strong></p>
<p>We have Enron email+financial (E+F) dataset. It contains 146 Enron managers to investigate. Each sample in this dictionary is containing 21 features. 18 people from this dataset labeled as POI. There are two imbalanced classes (many more non-POIs than POIs). Here's an example of one POI data point:</p>
LAY KENNETH L<br>
salary : 1072321<br>
to_messages : 4273<br>
deferral_payments : 202911<br>
total_payments : 103559793<br>
exercised_stock_options : 34348384<br>
bonus : 7000000<br>
restricted_stock : 14761694<br>
shared_receipt_with_poi : 2411<br>
restricted_stock_deferred: NaN<br>
total_stock_value : 49110078<br>
expenses : 99832<br>
loan_advances : 81525000<br>
from_messages : 36<br>
other : 10359729<br>
from_this_person_to_poi : 16<br>
poi : 1<br>
director_fees : NaN<br>
deferred_income : -300000<br>
long_term_incentive : 3600000<br>
email_address : kenneth.lay@enron.com<br>
from_poi_to_this_person : 123
<p><strong>Outliers</strong></p>
<p>Dataset contains some outliers. The TOTAL row is the biggest Enron E+F dataset outlier. We should remove it from dataset for reason it's a spreadsheet quirk. Moreover, there are 4 more outliers with big salary and bonus. Two people made bonuses more than 6 million dollars, and a salary of over 1 million dollars. There's no mistake in the data. Ken Lay and Jeffrey Skilling made such money. So, these data points should be left in and examine it with others.</p>
<p><strong>2. Feature selection process</strong></p>
<table>
<thead>
<tr class="header">
<th>Feature Selection</th>
<th>Justification</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Expenses</td>
<td>PoI (100%) had expenses as compared to non-poi (60%)</td>
</tr>
<tr class="even">
<td>Shared_receipt_with_poi</td>
<td>(14/18 :77%) had shared receipt with PoI as compared to non-poi (77/127:57%)</td>
</tr>
<tr class="odd">
<td>From_poi_to_this_person</td>
<td>Emails from poi to the employee that might have material information</td>
</tr>
</tbody>
</table>
<p><strong>New features</strong></p>
<p>In addition I create two new features which were considered in course:</p>
<table>
<thead>
<tr class="header">
<th>Feature Selection</th>
<th>Justification</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Total_payments/Salary</td>
<td>total_payments/salary to identify those who had most to gain from non-salaried compensation</td>
</tr>
<tr class="even">
<td>Total_stock_value_to_payments</td>
<td>Total_stock_value/total_payments to identify those who had most to gain from higher stock price</td>
</tr>
</tbody>
</table>
<p>Impact of New Features (using Decision Tree Classifier)</p>
<table>
<thead>
<tr class="header">
<th>Best scores without New Feature      |</th>
<th>Best scores with New Features</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><p>Accuracy: 0.82793</p>
<p>Precision: 0.36532</p>
<p>Recall: 0.39400</p>
<p>F1: 0.37912</p>
<p>F2: 0.38791</p></td>
<td><p>Accuracy: 0.84443</p>
<p>Precision: 0.43697</p>
<p>Recall: 0.30850</p>
<p>F1: 0.36166</p>
<p>F2: 0.32777</p></td>
</tr>
</tbody>
</table>

<p>Feature selection process included several iterations. On the first step I created set of features based on data visualization and intuition. Then I examine seven classifiers on these features, and optimized them using pipeline with selectKBest, pca and MinMaxScaler. Of these, Decision Tree gave the best accuracy, precision and recall with following feature importance:</p>
<p><strong>Feature Importance:</strong></p>
expenses 0.396<br>
shared_receipt_with_poi 0.271<br>
payment_to_salary 0.070<br>
total_stock_to_payments 0.263<br>
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,<br>
max_features=None, max_leaf_nodes=None,<br>
min_impurity_split=1e-07, min_samples_leaf=1,<br>
min_samples_split=2, min_weight_fraction_leaf=0.0,<br>
presort=False, random_state=None, splitter='best')<br>
Accuracy: 0.84443 Precision: 0.43697 Recall: 0.30850 F1: 0.36166 F2: 0.32777<br>
Total predictions: 14000 True positives: 617 False positives: 795 False negatives: 1383 True negatives: 11205
<p>Therefore, I chose the following features (dropping payment to salary) for the final run with following feature importance which improved precision to .53 and recall to .36! (see below for the full results)</p>
<p><strong>Selected Features:</strong></p>
<p>expenses</p>
<p>shared_receipt_with_poi</p>
<p>total_stock_to_payments</p>
<p><strong>3. Pick an algorithm</strong></p>
<p><strong>The following table describes all results of examination from the algorithm used:</strong></p>
<table>
<thead>
<tr class="header">
<th><strong>Algorithm</strong></th>
<th><strong>Pipeline</strong></th>
<th><strong>Accuracy</strong></th>
<th><strong>Precision</strong></th>
<th><strong>Recall</strong></th>
<th><strong>F1</strong></th>
<th><strong>F2</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><strong>Naive Bayes</strong></td>
<td>No</td>
<td>0.67838</td>
<td>0. 19497</td>
<td>0. 34850</td>
<td>0. 25004</td>
<td>0. 30108</td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0. 72746</td>
<td>0. 21792</td>
<td>0. 29800</td>
<td>0. 25174</td>
<td>0. 27760</td>
</tr>
<tr class="odd">
<td><strong>SVM</strong></td>
<td>No*</td>
<td></td>
</tr>
<tr class="even">
<td></td>
<td>Yes*</td>
<td></td>
</tr>
<tr class="odd">
<td><strong>Decision Tree</strong></td>
<td><strong>No</strong></td>
<td><strong>0. 85292 </strong></td>
<td><strong>0. 53235 </strong></td>
<td><strong>0. 36200</strong></td>
<td><strong>0. 43095 </strong></td>
<td><strong>0. 38675</strong></td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0. 82408</td>
<td>0. 37951</td>
<td>0. 22600</td>
<td>0. 28330</td>
<td>0. 24589</td>
</tr>
<tr class="odd">
<td><strong>Nearest Neighbors</strong></td>
<td>No</td>
<td>0.78369</td>
<td>0.18232</td>
<td>0.11650</td>
<td>0.14216</td>
<td>0.12557</td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0.78046</td>
<td>0.19759</td>
<td>0.13950</td>
<td>0.16354</td>
<td>0.14822</td>
</tr>
<tr class="odd">
<td><strong>Random Forest</strong></td>
<td>No</td>
<td>0.77738</td>
<td>0.28468</td>
<td>0.29550</td>
<td>0.28999</td>
<td>0.29327</td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0.77785</td>
<td>0.28937</td>
<td>0.30500</td>
<td>0.29698</td>
<td>0.30174</td>
</tr>
<tr class="odd">
<td><strong>AdaBoost</strong></td>
<td>No</td>
<td>0.81585</td>
<td>0.33852</td>
<td>0.20650</td>
<td>0.25652</td>
<td>0.22397</td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0.78877</td>
<td>0.24347</td>
<td>0.17700</td>
<td>0.20498</td>
<td>0.18722</td>
</tr>
<tr class="odd">
<td><strong>QDA</strong></td>
<td>No</td>
<td>0.67600</td>
<td>0.18237</td>
<td>0.31750</td>
<td>0.23167</td>
<td>0.27652</td>
</tr>
<tr class="even">
<td></td>
<td>Yes</td>
<td>0.67600</td>
<td>: 0.18237</td>
<td>0.31750</td>
<td>0.23167</td>
<td>0.27652</td>
</tr>
</tbody>
</table>
*Got a divide by zero when trying out: SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
<p><strong>Chosen algorithm</strong></p>
<p>Based on best performance level I picked Decision Tree as a final algorithm.</p>
<p><strong>4. Tune the algorithm</strong></p>
<p><strong>Reasons for algorithm tuning</strong></p>
<p>The main reason is to get better results from algorithm. I used <strong>GridSearchCV</strong> with following parameters to tune the algorithm.</p>
<table>
<thead>
<tr class="header">
<th><strong>Parameter</strong></th>
<th><p><strong>Settings for investigation</strong></p></th>
<th><strong>Best Value</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>min_samples_split</td>
<td>[2,4,6,8]</td>
<td>2</td>
</tr>
<tr class="even">
<td>splitter</td>
<td>['random','best']</td>
<td>best</td>
</tr>
<tr class="odd">
<td>max_depth</td>
<td>[2,4,6,8,10,15]</td>
<td>4</td>
</tr>
<tr class="even">
<td>criterian</td>
<td>['gini','entropy']</td>
<td>'entropy'</td>
</tr>
</tbody>
</table>
<p><strong>5. Validation</strong></p>
<p>To validate my analysis I used <a href="http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html">stratified shuffle split cross validation </a> developed by Udacity and defined in tester.py file. I had to modify test_classifier to return all the computed metrics for comparison with prevailing values. In addition, the input arrays to fit function had to numpy arrays for the pipeline classifier.</p>
<p><strong>6. Evaluation metrics</strong></p>
<p>I used precision and recall evaluation metrics to estimate model. Final results can be found in table below</p>
<table>
<thead>
<tr class="header">
<th><strong>Accuracy</strong></th>
<th><strong>Precision</strong></th>
<th><strong>Recall</strong></th>
<th><strong>F1</strong></th>
<th><strong>F2</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<th>0.88100</th>
<th>0.71228</th>
<th>0.38</th>
<th>0.49560</th>
<th>0.41910</th>
</tr>
<tr class="even">
<th><strong>True Positive</strong></th>
<th><strong>False Positive</strong></th>
<th><strong>False Negative</strong></th>
<th><strong>True Negatives</strong></th>
<th><strong>Total</strong></td>
</tr>
<tr class="odd">
<th>760</th>
<th>307</th>
<th>1240</th>
<th>10693</th>
<th>13000</th>
</tr>
</tbody>
</table>
<p><strong>Conclusion</strong></p>
<p>With Precision of .71 and Recall of .38, project goal of higher than .3 was reached. In this example, higher precision is more important as we want to minimize innocent employees identified as poi suspects. At the same time Recall = 0.38 says only 38% of all POIs were identified.</p>
<p>We have very imbalanced classes in E+F dataset. In addition, almost half of all POIs weren't included in dataset. Under the circumstances, the result received is quite good though it's not ideal, of course.</p>
