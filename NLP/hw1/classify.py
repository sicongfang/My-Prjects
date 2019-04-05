#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:15:28 2017

@author: sicongfang
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB

# Load training data 
import re 
#print(sys.argv)
train_file = open(sys.argv[1], "r")
train_lines = train_file.readlines()
train_data=[]
train_label = []
# Extract label and remove links in each tweet
for i in train_lines:
    temp=re.search('\t\w+\n',i).group(0).replace('\t','').replace('\n','')
    train_label.append(temp)
    temp2=re.sub(r'http.+\t','',i)
    temp2=re.sub(temp,'',temp2)
    train_data.append(temp2.replace('\n',''))

# Load Validation data

dev_file = open(sys.argv[2], "r")
dev_lines = dev_file.readlines()
dev_data=[]
dev_label = []
# Extract label and remove links in each tweet
for i in dev_lines:
    temp=re.search('\t\w+\n',i).group(0).replace('\t','').replace('\n','')
    dev_label.append(temp)
    temp2=re.sub(r'http.+\t','',i)
    temp2=re.sub(temp,'',temp2)
    dev_data.append(temp2.replace('\n',''))
    
from sklearn.preprocessing import LabelEncoder
labels=LabelEncoder()
y_train=labels.fit_transform(train_label)
y_test = labels.transform(dev_label)

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

vect_emoji = CountVectorizer(ngram_range=(1, 1),
                       lowercase=False,
                       preprocessor=lambda x: x,
                       token_pattern = r'\W'
                      )

X_train_emoji = vect_emoji.fit_transform(train_data)
X_test_emoji = vect_emoji.transform(dev_data)

# adding stop words, min_df...
vect = CountVectorizer(ngram_range=(1,2),
                       min_df=2,
                       stop_words="english")
#                        ,
#                       token_pattern = r'\b\w[\w\']+\b')
X_train_0 = vect.fit_transform(train_data)
X_test_0 = vect.transform(dev_data)

import scipy
#train_count = scipy.sparse.csr_matrix(np.column_stack((np.array(X_train_rep),np.array(X_train_ep))))
# X_train = scipy.sparse.hstack((X_train_0, X_train_cap,X_train_emoji,train_count))
X_train = scipy.sparse.hstack((X_train_0, X_train_emoji))



#dev_count = scipy.sparse.csr_matrix(np.column_stack((np.array(X_test_rep),np.array(X_test_ep))))
X_test = scipy.sparse.hstack((X_test_0, X_test_emoji))

best=MultinomialNB()
x=best.fit(X_train,y_train)

print("Training Accuracy:", x.score(X_train,y_train))
print("Test Accuracy:", x.fit(X_train,y_train).score(X_test,y_test))

from sklearn.pipeline import make_pipeline, Pipeline,FeatureUnion

pipe = Pipeline([
       ('features', FeatureUnion([
                    ('n-gram:', vect),
                    ('emoji:', vect_emoji)
                ])
        
        )])
feature = np.array(pipe.named_steps['features'].get_feature_names())
index=np.argsort(np.abs(best.coef_))
top20=feature[index[0][-20:]]

import pickle 
filename='model.pkl'
filehandler = open(filename, 'wb') 
pickle.dump(x, filehandler) 
y_test=scipy.sparse.csr_matrix(y_test.reshape((-1,1)))
test = scipy.sparse.hstack((X_test,y_test))
filename='test'
filehandler2 = open(filename, 'wb') 
pickle.dump(test, filehandler2) 
pickle.dump(top20,filehandler2)
