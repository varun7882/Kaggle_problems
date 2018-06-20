# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 01:25:54 2018

@author: VaSrivastava
"""

import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

data_X=pd.read_csv('normalizedX.csv')
data_X.insert(0,'bias',1)
test_data=pd.read_csv('normalized_testX.csv')
test_data.insert(0,'bias',1)
X_test=test_data.values
data_y=pd.read_csv('targetY.csv')
X=data_X.values
y=data_y.values

tf.