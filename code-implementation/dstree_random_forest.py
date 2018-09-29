# -*- coding: utf-8 -*-

'''
Created on 2018. 9. 24.

@author: jason96

Apply Gini Impurity
'''

import pandas as pd
from graphviz import Digraph
import os
import operator
import numpy as np

raw_data = {
    'name': ["Kang", "Kim", "Choi", "Park", "Yoon"],
    '짱절미': [True, False, False, False, False],
    '셀스타그램': [False, False, True, False, False],
    'like': [True, False, True, True, False]
}

pd_data = pd.DataFrame(raw_data)
pd_data = pd_data.set_index("name")
label_name = "like"
feature_names = pd_data.columns.difference([label_name])


def display_node(dot, key, node):
    if node["leaf"] is True:
        proba = node['proba']
        proba = round(proba, 4)
        proba = str(proba)

        dot.node(key, proba)
    else:
        desc = node['desc']
        dot.node(key, desc)

        if "left" in node:
            left_key = key + "L"
            display_node(dot, left_key, node['left'])
            dot.edge(key, left_key)

        if "right" in node:
            right_key = key + "R"
            display_node(dot, right_key, node['right'])
            dot.edge(key, right_key)

    dot.render('graphviz-files/dstree.gv', view=True)


def display_tree(tree):
    dot = Digraph(comment='Decision Tree')
    display_node(dot, "Root", tree)


def predict(data, node):
    if node['leaf']:
        proba = node["proba"]
        result = dict(zip(data.index, len(data) * [proba]))
    else:
        rule = node['rule']

        left_data = data[rule(data)]
        left_result = predict(left_data, node['left'])

        right_data = data[~rule(data)]
        right_result = predict(right_data, node['right'])

        return {**left_result, **right_result}

    return result


def binary_rule(data, feature_name, value):
    return data[feature_name] == value


def make_rule(method, feature_name, value):
    def call_condition(data):
        return method(data, feature_name, value)
    return call_condition


def make_rules(feature_names):
    rules = {}
    feature_names = list(feature_names)
    for feature_name in feature_names:
        rules[feature_name] = make_rule(binary_rule, feature_name, True)
    return rules


def get_best_rule(data, rules):

    gini_indexes = {}
    for feature_name, rule in rules.items():
        true_data = data[rule(data)]
        true_proba = true_data[label_name].mean()
        false_proba = 1 - true_proba
        gini_index = true_proba*(1-false_proba) - false_proba*(1-true_proba)
        gini_indexes[feature_name] = gini_index

    sorted_x = sorted(gini_indexes.items(), key=operator.itemgetter(1))

    for k, v in sorted_x:  # @UnusedVariable
        return k, rules[k]


def make_node(data, rules):

    if len(rules) > 0:

        feature_name, rule = get_best_rule(data, rules)

        left_data = data[rule(data)]
        right_data = data[~rule(data)]

        if len(left_data) > 0 and len(right_data) > 0:
            del rules[feature_name]
            node = {'leaf': False, 'desc': feature_name, 'rule': rule}
            node['left'] = make_tree(left_data, rules.copy())
            node['right'] = make_tree(right_data, rules.copy())
            return node

    proba = data[label_name].mean()
    node = {'leaf': True, 'proba': proba}
    return node


def make_tree(data, feature_names):

    rules = make_rules(feature_names)
    return make_node(data, rules)


def display_predict(predict):
    for k, v in predict.items():
        print(k, v)


def bootstrap(data, feature_names, label_name):
    feature_data = data[feature_names]
    num_rows, num_cols = feature_data.shape
    index = np.random.choice(feature_data.index, size=num_rows, replace=True)

    if max_feature == None:  # @IgnorePep8
        num_cols = np.sqrt(num_cols)
    else:
        num_cols = num_cols * max_feature
    num_cols = int(num_cols)

    columns = np.random.choice(feature_data.columns, size=num_cols,
                               replace=False)
    # If index and columns are specified,
    # a new table is created based on the values.
    result = feature_data.loc[index, columns]
    result[label_name] = data[label_name]

    return result


def make_forest(data):
    forest = []
    for _ in range(n_estimators):
        bootstrapped_data = bootstrap(data, feature_names, label_name)
        bs_feature_names = bootstrapped_data.columns.difference([label_name])
        tree = make_tree(bootstrapped_data, bs_feature_names)
        forest.append(tree)
    return forest


def predict_forest(data, forest):
    prediction_total = []

    for tree in forest:
        prediction = predict(data, tree)
        prediction = pd.Series(prediction)
        prediction_total.append(prediction)

    prediction_total = pd.concat(prediction_total, axis=1, sort=False)

    prediction_total = prediction_total.mean(axis=1)

    return prediction_total


if __name__ == '__main__':

    max_feature = None
    n_estimators = 10
    os.environ["PATH"] += os.pathsep + '/usr/local/bin'

    forest = make_forest(pd_data)
    display_predict(predict_forest(pd_data, forest))

    # tree = make_tree(pd_data, rules)
    # display_tree(tree)
    # display_predict(predict(pd_data, tree))
