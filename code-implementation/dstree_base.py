# -*- coding: utf-8 -*-

'''
Created on 2018. 9. 24.

@author: jason96

Base Decision Tree
'''

import pandas as pd
from graphviz import Digraph
import os

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


def get_best_rule(rules):

    for feature_name, rule in rules.items():
        return feature_name, rule


def make_tree(data, rules):
    if len(rules) > 0:
        feature_name, rule = get_best_rule(rules)

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


def display_predict(predict):
    for k, v in predict.items():
        print(k, v)


if __name__ == '__main__':

    os.environ["PATH"] += os.pathsep + '/usr/local/bin'
    rules = make_rules(feature_names)
    tree = make_tree(pd_data, rules)
    # display_tree(tree)
    display_predict(predict(pd_data, tree))
