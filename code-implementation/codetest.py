# -*- coding: utf-8 -*-

'''
Created on 2018. 9. 29.

@author: jason96
'''

import pandas as pd


def codetest():

    data = {
        'name': ["Kang", "Kim", "Choi", "Park", "Yoon"],
        '짱절미': [True, False, False, False, False],
        '셀스타그램': [False, False, True, False, False],
        'like': [True, False, True, True, False]
    }

    data = pd.DataFrame(data)
    data = data.set_index("name")

    left_data = data[data['짱절미'] == True] # @IgnorePep8 @NoEffect


if __name__ == '__main__':
    codetest()
