#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:31:42 2017

@author: sicongfang
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Load training data 
import re 
train_file = open("train_newline.txt", "r")
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
dev_file = open("dev_newline.txt", "r")
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
y_train1=labels.fit_transform(train_label)
y_test1 = labels.transform(dev_label)

# Best model
filehandler = open("test", 'rb') 
test_data = pickle.load(filehandler) 
top20 = pickle.load(filehandler)
test=test_data.tocsr()
X_test = test[:,0:-1]
y_test = test[:,-1]

filehandler2 = open("model.pkl",'rb')
model = pickle.load(filehandler2)
y_pred = model.predict(X_test)
print("------------------------------------------------------------------------")
print("Best Model")
print("Top20 features:")
print(list(top20))
print("Accuracy:",accuracy_score(y_test.toarray().reshape((-1,)),y_pred))
print("contingency table:")
print(confusion_matrix(y_test.toarray().reshape((-1,)),y_pred))

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
#------------------------------------------------------------------------------
## Uni-gram
vect2 = CountVectorizer(ngram_range=(1, 1),
                        stop_words="english",
                        min_df=2
                      )
X_train1 = vect2.fit_transform(train_data)
X_test1 = vect2.transform(dev_data)
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

#Logsitic Regression
# cv3=LogisticRegression()
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
#              'fit_intercept':[True,False]}
# grid3 = GridSearchCV(cv3, param_grid, cv=5)
# print('uni-gram, logistic Regression:')
# print(grid3.fit(X_train1,y_train).score(X_test1,y_test))

# #SVM
# from sklearn.svm import LinearSVC
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 3, 5, 10]
# #               ,'kernel': ['linear','rbf','poly']
#              }
# uni_svm = LinearSVC()
# uni_svm_grid = GridSearchCV(uni_svm, param_grid, cv=5)
# print('uni-gram, SVM:')
# print(uni_svm_grid.fit(X_train1,y_train).score(X_test1,y_test))

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
uni_NB = MultinomialNB().fit(X_train1,y_train1)
y_pred1=uni_NB.predict(X_test1)
feature = np.array(vect2.get_feature_names())
index=np.argsort(np.abs(uni_NB.coef_))
top20=feature[index[0][-20:]]

print("------------------------------------------------------------------------")
print('uni-gram, Naive Bayes:')
print("Training Accuracy:",uni_NB.score(X_train1,y_train1))
print("Test Accuracy:",accuracy_score(y_test1,y_pred1))
print("Top20 Features:")
print(list(top20))
print("Contingency Matrix;")
print(confusion_matrix(y_test1,y_pred1))


#------------------------------------------------------------------------------
#bi-gram
vect_bi = CountVectorizer(ngram_range=(2, 2))
X_train_bi = vect_bi.fit_transform(train_data)
X_test_bi = vect_bi.transform(dev_data)

# #Logsitic Regression
# cv3=LogisticRegression()
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
#              'fit_intercept':[True,False]}
# grid3 = GridSearchCV(cv3, param_grid, cv=5)
# print('bi-gram, logistic Regression:')
# print(grid3.fit(X_train_bi,y_train).score(X_test_bi,y_test))

# from sklearn.svm import LinearSVC
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 3, 5, 10]
# #               ,'kernel': ['linear','rbf','poly']
#              }
# uni_svm = LinearSVC()
# uni_svm_grid = GridSearchCV(uni_svm, param_grid, cv=5)
# print('bi-gram, SVM:')
# print(uni_svm_grid.fit(X_train_bi,y_train).score(X_test_bi,y_test))

# Naive Bayes

bi_NB = MultinomialNB().fit(X_train_bi,y_train1)
y_pred1=bi_NB.predict(X_test_bi)
feature = np.array(vect_bi.get_feature_names())
index=np.argsort(np.abs(bi_NB.coef_))
top20=feature[index[0][-20:]]
print("------------------------------------------------------------------------")
print('bi-gram, Naive Bayes:')
print("Training Accuracy:",bi_NB.score(X_train_bi,y_train1))
print("Test Accuracy:",accuracy_score(y_test1,y_pred1))
print("Top20 Features:")
print(list(top20))
print("Contingency Matrix;")
print(confusion_matrix(y_test1,y_pred1))


#------------------------------------------------------------------------------
#tri-gram
vect_tri = CountVectorizer(ngram_range=(3, 3))
X_train_tri = vect_tri.fit_transform(train_data)
X_test_tri = vect_tri.transform(dev_data)

# #Logsitic Regression
# cv3=LogisticRegression()
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
#              'fit_intercept':[True,False]}
# grid3 = GridSearchCV(cv3, param_grid, cv=5)
# print('tri-gram, logistic Regression:')
# print(grid3.fit(X_train_tri,y_train).score(X_test_tri,y_test))

# from sklearn.svm import LinearSVC
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 3, 5, 10]
# #               ,'kernel': ['linear','rbf','poly']
#              }
# uni_svm = LinearSVC()
# uni_svm_grid = GridSearchCV(uni_svm, param_grid, cv=5)
# print('tri-gram, SVM:')
# print(uni_svm_grid.fit(X_train_tri,y_train).score(X_test_tri,y_test))

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
tri_NB = MultinomialNB().fit(X_train_tri,y_train1)
y_pred1=tri_NB.predict(X_test_tri)
feature = np.array(vect_tri.get_feature_names())
index=np.argsort(np.abs(tri_NB.coef_))
top20=feature[index[0][-20:]]

print("------------------------------------------------------------------------")
print('tri-gram, Naive Bayes:')
print("Training Accuracy:",tri_NB.score(X_train_tri,y_train1))
print("Test Accuracy:",accuracy_score(y_test1,y_pred1))
print("Top20 Features:")
print(list(top20))
print("Contingency Matrix;")
print(confusion_matrix(y_test1,y_pred1))



#------------------------------------------------------------------------------
#Appendix
#adding emoji features
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import scale, StandardScaler
# from sklearn.feature_extraction.text import CountVectorizer

# vect_emoji = CountVectorizer(ngram_range=(1, 1),
#                        lowercase=False,
#                        preprocessor=lambda x: x,
#                        token_pattern = r'\W'
#                       )

# X_train_emoji = vect_emoji.fit_transform(train_data)
# X_test_emoji = vect_emoji.transform(dev_data)

# # adding stop words, min_df...
# vect = CountVectorizer(ngram_range=(1,2),
#                        min_df=2,
#                        stop_words="english")
# #                        ,
# #                       token_pattern = r'\b\w[\w\']+\b')
# X_train_0 = vect.fit_transform(train_data)
# X_test_0 = vect.transform(dev_data)

# # all caps

# vect_cap = CountVectorizer(ngram_range=(1,1),
#                           lowercase = False,
#                           preprocessor=lambda x: x,
#                           token_pattern = r'[A-Z][A-Z]+')

# X_train_cap = vect_cap.fit_transform(train_data)
# X_test_cap = vect_cap.transform(dev_data)

# repeating words
# X_train_rep = []
# for i in train_data:
#     temp=re.sub(r'[^A-Za-z0-9]+', ' ',i)
#     X_train_rep.append(len(i)-len(temp))               
# X_test_rep = []
# for i in dev_data:
#     temp=re.sub(r'[^A-Za-z0-9]+', ' ',i)
#     X_test_rep.append(len(i)-len(temp))  
    
# # multiple exclaimation point
# X_train_ep= []
# for i in train_data:
#     temp=re.search(r'[A-Z]+ \!{1,}',i)
#     if temp == None:
#         X_train_ep.append(0)
#     else:
#         X_train_ep.append(len(temp.group(0)))              
# X_test_ep = []
# for i in dev_data:
#     temp=re.search(r'[A-Z]+ \!{1,}',i)
#     if temp ==None:
#         X_test_ep.append(0)
#     else:
#         X_test_ep.append(len(temp.group(0)))        

#import scipy
# #train_count = scipy.sparse.csr_matrix(np.column_stack((np.array(X_train_rep),np.array(X_train_ep))))
# # X_train = scipy.sparse.hstack((X_train_0, X_train_cap,X_train_emoji,train_count))
# X_train = scipy.sparse.hstack((X_train_0, X_train_emoji))



# #dev_count = scipy.sparse.csr_matrix(np.column_stack((np.array(X_test_rep),np.array(X_test_ep))))
# X_test = scipy.sparse.hstack((X_test_0, X_test_emoji))











