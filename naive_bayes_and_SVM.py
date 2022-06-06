#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:41:08 2022

@author: lizziezhang
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#如何传入label

#split training set and text set
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=12345)

#Bernoulli Naive Bayes with evaluation
bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print("Accuracy: ")
print(round(accuracy_score(y_test,y_pred)*100,2),"%")

#Linear SVM with evaluation
linearsvc_clf = LinearSVC()  
linearsvc_clf.fit(X_train, y_train) 
predict_y = linearsvc_clf.predict(X_test) 

print(confusion_matrix(y_test,predict_y))  
print(classification_report(y_test,predict_y))
print("Accuracy: ")
print(round(accuracy_score(y_test,predict_y)*100,2),"%")