#!/usr/bin/python

import sys
import pickle
import random
import numpy
import matplotlib
import matplotlib.pyplot as plt
import helper_functions
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tester import dump_classifier_and_data, test_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


### Choose the features to use
output_label = 'poi'
email_features_list = [
    #'email_address', """depricated because of error in tester.py"""
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]
features_list = [output_label] + financial_features_list + email_features_list
print "Length:", len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "-------------------------TEST THE DEFAULT DATASET AND FEATURE LIST------------------"
default_list = [output_label] + financial_features_list + email_features_list
with open("final_project_dataset.pkl", "r") as data_file:
    default_data = pickle.load(data_file)

data = featureFormat(default_data, default_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
     features, labels, test_size=0.3, random_state=42)

### Test Using Decision Tree Classifier
clf_default = DecisionTreeClassifier()
clf_default.fit(features_train, labels_train)
pred = clf_default.predict(features_test)
accuracy = clf_default.score(features_test, labels_test)
print "Decision Tree accuracy:", accuracy

clf_dt_default = DecisionTreeClassifier()
test_classifier(clf_dt_default, default_data, default_list)


print "--------------------------DATASET EXPLORATION---------------------------------------"
helper_functions.enron_data_summary(data_dict)


print "--------------------------DATASET CLEANING---------------------------------------"
### Find outliers using visualization

#helper_functions.outliers_visualization(data_dict, "salary", "bonus")

### Remove outliers from the dataset and check for additional outliers

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
data_dict.pop('LOCKHART EUGENE E',0)

#helper_functions.outliers_visualization(data_dict, "salary", "bonus")

### Validate that the outliers were removed. The total # of rows should have decreased by 3

print "# of People in the Dataset, minus the outliers:", len(data_dict)

### Find additional outliers

for employee in data_dict:
    if (data_dict[employee]["salary"] != 'NaN') and (data_dict[employee]["bonus"] != 'NaN'):
        if float(data_dict[employee]["salary"]) >1000000:
            print "people with salary over 1 million:", employee

for employee in data_dict:
    if float(data_dict[employee]["bonus"]) >7000000:
        print "people with bonus over 1 million:", employee

### Find all the NaN values and replace them with zeros

def find_nan_values():
    # Update NaN values with 0 except for email address
    people_keys = data_dict.keys()
    feature_keys = data_dict[people_keys[0]]
    nan_features = {}
    # Get list of NaN values and replace them
    for feature in feature_keys:
        nan_features[feature] = 0
    for person in people_keys:
        for feature in feature_keys:
            if data_dict[person][feature] == 'NaN':
                nan_features[feature] += 1
    print "# of Missing Values per Feature:",
    for feature in feature_keys:
        print feature, nan_features[feature]

find_nan_values()


def fill_nan_values():
    # Update NaN values with 0 except for email address
    people_keys = data_dict.keys()
    feature_keys = data_dict[people_keys[0]]
    nan_features = {}
    # Get list of NaN values and replace them
    for feature in feature_keys:
        nan_features[feature] = 0
    for person in people_keys:
        for feature in feature_keys:
            if feature != 'email_address' and \
                data_dict[person][feature] == 'NaN':
                data_dict[person][feature] = 0
                nan_features[feature] += 1

    return nan_features

fill_nan_values()

find_nan_values()

# Save the cleaned data to a new dataset called my_dataset
my_dataset = data_dict

# print "----------------CHECKING INFLUENCE OF NEW FEATURES and K-VALUES------------"
# kbest_default = helper_functions.get_k_best(my_dataset, features_list, 12)


# ### Add the top features with the POI label, creating a new feature list
# features_list_default = [output_label] + kbest_default.keys()

# clf_dt = DecisionTreeClassifier()
# test_classifier(clf_dt, my_dataset, features_list)


print "--------------------------CREATE NEW FEATURES-----------------------------"

### Using the new cleaned dataset, create two new features
### New Feature #1: POI Interaction
helper_functions.poiFraction(my_dataset,features_list)

### New Feature #2: Sum of all money earned, present and future earnings
helper_functions.total_money_earned(my_dataset,features_list)

print "Features List With Newly Created Features:", features_list
print "Length:", len(features_list)
print "Check dataset with new features:", my_dataset['LAY KENNETH L']



print "--------------------------SELECT TOP FEATURES-----------------------------------------"
### Decide which features are the most important using SelectKBest from scikit-learn
kbest = helper_functions.get_k_best(my_dataset, features_list, 10)


### Add the top features with the POI label, creating a new feature list
features_list = [output_label] + kbest.keys()
print "Refined Features List:", features_list
print "Length:", len(features_list)


print "--------------------------PRELIMINARY TESTING CLASSIFIERS-----------------------------"
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
     features, labels, test_size=0.3, random_state=42)

### Test Using Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print "Decision Tree accuracy:", accuracy

# ### compute precision score and recall score
print "Precision Score:", precision_score(labels_test, pred)
print "Recall Score:", recall_score(labels_test, pred)

target_names = ["Not POI", "POI"]

print '\n Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)


clf_dt = DecisionTreeClassifier()
test_classifier(clf_dt, my_dataset, features_list)


# print "--------------------------TUNING THE CLASSIFIER-----------------------------------------"
# ### Tune the Decision Tree classifier using GridSearchCV

# # dtree = DecisionTreeClassifier()
# # best_parameters = helper_functions.classifier_tuning_pipeline(dtree, 10, features_list, my_dataset)
# # print 'Best parameters: ', best_parameters.best_params_


print "--------------------------EVALUATION-----------------------------------------"
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf_dt = DecisionTreeClassifier(criterion='gini', max_depth = 6, min_samples_leaf = 1, min_samples_split = 2)
test_classifier(clf_dt, my_dataset, features_list)

dump = dump_classifier_and_data(clf_dt, my_dataset, features_list)
print dump
