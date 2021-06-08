#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:07:48 2021

@author: jadelrez
"""

import pandas as pd
import seaborn as sns
df= pd.read_csv('loan.csv')

#drop id column 
df=df.drop('Loan_ID', axis=1)
#checking and dropping missing values
sns.heatmap(df.isnull())
df= df.dropna()

#Convert categorical data into dummy variables 
df=pd.get_dummies(df, drop_first= True)

#split into x and y
x=df.drop('Loan_Status_Y', axis = 1)
y = df['Loan_Status_Y']

#Train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=41) 

#Build decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

#Evaluate decision tree on training set
from sklearn.metrics import f1_score
y_pred_train= dt.predict(x_train)
f1_score(y_train,y_pred_train)

#testing set
y_pred_test= dt.predict(x_test)
f1_score(y_test,y_pred_test)

#Lets plot the tree
from sklearn.tree import plot_tree
plot_tree(dt)
#make tree look better
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


#Max depth of the tree
dt.tree_.max_depth

#Using gridsearch to improve tree preformance 
parameter_grid = {'max_depth':range(2,16),'min_samples_split':range(2,6)}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dt,parameter_grid, verbose = 3,scoring= 'f1')

#Now fit the grid 
grid.fit(x_train,y_train)

#Checking the best parameters
grid.best_params_ 
'''{'max_depth': 2, 'min_samples_split': 2}'''

#Using the optimized parameters to evaluate for performance improvmeent 
dt=DecisionTreeClassifier(max_depth = 2, min_samples_split=2)
dt.fit(x_train,y_train)

#Evaluate decision tree on training set
from sklearn.metrics import f1_score
y_pred_train= dt.predict(x_train)
f1_score(y_train,y_pred_train)
print('training score is',f1_score(y_train,y_pred_train))

# Evaluatee on testing set
y_pred_test= dt.predict(x_test)
f1_score(y_test,y_pred_test)
print('testing score is ',f1_score(y_test,y_pred_test))



###plot the optimized tree 
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

#####################
####
#Random Forests 

from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators = 500, max_depth = 2, min_samples_split=2)

rfc.fit(x_train,y_train)

##Hoow it performoors on training and testing
y_pred_train= rfc.predict(x_train)
f1_score(y_train,y_pred_train)
print('training score is',f1_score(y_train,y_pred_train))

# Evaluatee on testing set
y_pred_test= rfc.predict(x_test)
f1_score(y_test,y_pred_test)
print('testing score is ',f1_score(y_test,y_pred_test))


#Initialize grid search on random forest o improve the model?

parameter_grid = {'max_depth':range(2,16),'min_samples_split':range(2,6)}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(rfc,parameter_grid, verbose = 3,scoring= 'f1')

#Now fit the grid 
grid.fit(x_train,y_train)

#Checking the best parameters
grid.best_params_ 
































































