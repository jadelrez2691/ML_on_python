#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:15:23 2021

@author: jadelrez
"""

import pandas as pd 

df = pd.read_csv("baseball.csv")

#seperate x and y values
x = df[['Hits','Years']]
y= df['Salary']

#Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =1,
                                                test_size = 0.3)

#Get the model
from sklearn.tree import DecisionTreeRegressor 
dtr= DecisionTreeRegressor()
dtr.fit(x_train,y_train)

#Make predictions
y_pred_dtr =dtr.predict(x_test)

##evalualte model 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred_dtr)
rmse = mse ** 0.5
print(rmse)

#Visualize the tree
from sklearn.tree import plot_tree
plot_tree(dtr)

dtr.tree_.max_depth


parameter_grid = {'max_depth':range(2,14),'min_samples_split':range(2,6)}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dtr,parameter_grid, verbose = 3)

grid.fit(x_train, y_train)

#Now fit the grid 
grid.best_params_ 
''' {'max_depth': 2, 'min_samples_split': 4}'''


dtr= DecisionTreeRegressor(max_depth = 2 , min_samples_split = 4)
dtr.fit(x_train,y_train)

#Make predictions
y_pred_dtr =dtr.predict(x_test)

##evalualte model 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred_dtr)
rmse = mse ** 0.5
print(rmse)














