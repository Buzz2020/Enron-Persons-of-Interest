#!/usr/bin/python

import sys
import pickle
import random
import numpy
import matplotlib
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def enron_data_summary(data):
    """
    find: total # of people in the dataset, # of total features
    per person, # of POIs, # of NaN values per feature,
    # of people with NaN values, # of POIs with NaN for
    total_payments feature
    """
    # Find the # of people in the dataset where the key value is the person's name
    print "# of People in the Dataset:", len(data)

    # Find the number of features per person
    features_per_person = data['LAY KENNETH L'].keys()
    print "# of Total Features Per Person:", len(features_per_person)

    # find the number of persons of interest
    poi_count = 0
    for key, value in data.items():
        if value["poi"] == True:
            poi_count += 1
    print "# of Predefined POIs:", poi_count

    # print list of POIs
    """
    used to get a quick glance of the list of POIs
    """
    # for key, value in data.items():
    #     if value["poi"] == True:
    #         print "List of POIs:", key

    # how many people in the dataset have "NaN" as their total payments?
    nan_total_payments_count = 0
    for key, value in data.items():
        if value["total_payments"] == 'NaN':
            nan_total_payments_count += 1
    print "People With a Blank 'Total Payments' Feature:", nan_total_payments_count

    # how many POIs in the dataset have "NaN" as their total payments?
    poi_nan_total_payments_count = 0
    for key, value in data.items():
        if value["poi"] == True and value["total_payments"] == 'NaN':
            poi_nan_total_payments_count += 1
    print "POIs With a Blank 'Total Payments' Feature", poi_nan_total_payments_count

def outliers_visualization(data, x_value, y_value):
    """
    creates a scatterplot to help discover outliers, visually
    """
    scatterplot_array = featureFormat(data, [x_value, y_value])

    for point in scatterplot_array:
        x_axis = point[0]
        y_axis = point[1]
        matplotlib.pyplot.scatter( x_axis, y_axis )

    matplotlib.pyplot.xlabel("x_value")
    matplotlib.pyplot.ylabel("y_value")
    #plt.savefig('salary_bonus_scatterplot.png')
    matplotlib.pyplot.show()

def poiFraction(data_dict, features_list):
    for key, value in data_dict.iteritems():
        value["poi_interaction"] = 0
        if value["to_messages"] and value["from_this_person_to_poi"] != 0:
            value["total_messages"] = value["to_messages"] + value["from_messages"]
            value["poi_messages"] = value["from_poi_to_this_person"] + value["from_this_person_to_poi"]
            value["poi_interaction"] = float(value["poi_messages"]) / float(value["total_messages"])
        else:
            value["poi_interaction"] == 'NaN'
    features_list += ["poi_interaction"]

def total_money_earned(data_dict, features_list):
    """ mutates data dict to add aggregate values from stocks and salary """
    fields = ['total_stock_value', 'exercised_stock_options', 'salary']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['total_money_earned'] = sum([person[field] for field in fields])
        else:
            person['total_money_earned'] = 'NaN'
    features_list += ['total_money_earned']


def get_k_best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features

def classifier_tuning_pipeline(classifier_type, kbest, features_list, my_dataset):
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    shuffle_split = StratifiedShuffleSplit(labels, 100, random_state=42)

    kbest = SelectKBest(k=kbest)
    scaler = MinMaxScaler()
    classifier = classifier_type ### come back to this and match with example2 ###
    pipeline = Pipeline(steps=[('minmax_scaler', scaler), ('feature_selection', kbest), ("classify", classifier)])

    parameters= {'classify__min_samples_leaf': [1,2,3,4,5],
             'classify__max_depth': [2,4,6,8,10],
             'classify__criterion': ['gini', 'entropy'],
             'classify__min_samples_split': [2,4,6,8,10]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=shuffle_split)
    cv.fit(features, labels)
    return cv
