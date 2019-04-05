#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:15:29 2017

@author: sicongfang
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

filehandler = open(sys.argv[2], 'rb') 
test_data = pickle.load(filehandler) 
top20 = pickle.load(filehandler)
test=test_data.tocsr()
X_test = test[:,0:-1]
y_test = test[:,-1]

filehandler2 = open(sys.argv[1],'rb')
model = pickle.load(filehandler2)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("Top20 features:")
print(list(top20))
print("Accuracy:",accuracy_score(y_test.toarray().reshape((-1,)),y_pred))
print("contingency table:")
print(confusion_matrix(y_test.toarray().reshape((-1,)),y_pred))
#print(y_test.toarray().reshape((-1,)).shape)

