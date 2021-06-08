#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 07:58:31 2021

@author: jadelrez
"""

import pandas as pd 

df = pd.read_csv("baseball-1.csv")

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

#Visaluzai the tree
from sklearn.tree import plot_tree
plot_tree(dtr)

#Inclass asignement : tune this tree using grid search 

#Doing random forests 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 500)
rfr.fit(x_train,y_train)

#Make predictions
y_pred_rfr = rfr.predict(x_test)

#Evaluate th emodel 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred_rfr)
rmse = mse ** 0.5
print(rmse)


##############################################################################


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('cancer-2.csv')

x=df.drop('target', axis = 1)
y= df['target']

model1= LogisticRegression(solver= 'liblinear')
model2 =SVC()
model3= DecisionTreeClassifier()
model_combo =VotingClassifier(estimators = [('lrp',model1),('sv', model2),
                                            ('dt', model3)])


#using cross validation to evaluate the model 
model1_score = cross_val_score(model1,x,y, scoring = 'f1',cv=10, verbose = 3) 
print('the score is', model1_score.mean())
#0.96

#score for model 2
model2_score = cross_val_score(model2,x,y, scoring = 'f1',cv=10, verbose = 3) 
print('the score is', model2_score.mean())
#0.93

#score for model 3
model3_score = cross_val_score(model3,x,y, scoring = 'f1',cv=10, verbose = 3) 
print('the score is', model3_score.mean())
#0.92

##combination model
model_combo_score = cross_val_score(model_combo,x,y, scoring = 'f1',cv=10, 
                                    verbose = 3) 
print('the score is', model_combo_score.mean())
#0.966

####################    BOOSTING  ############################################

from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('cancer-2.csv')
x=df.drop('target', axis = 1)
y= df['target']

###train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =1,
                                                test_size = 0.3)

#Get the ada boost
adb= AdaBoostClassifier(n_estimators= 100)
adb.fit(x_train,y_train)

#Make predictions
y_pred = adb.predict(x_test)

#Get the f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


#Can you get f1 using cross validation 
# use cv= 10
from sklearn.model_selection import cross_val_score
adb_score = cross_val_score(adb,x,y,scoring = 'f1', cv= 10 ,verbose = 3)
adb_score.mean()


#################### KNN ############################################


df= pd.read_csv('heart-1.csv')


#We will go with min max scaler method sicne the distributions are not normal

#Separate into x and y
y=df['target']
x=df.drop('target', axis = 1)

#Step 1: split into traina dn test sets
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state =101,
                                                test_size = 0.3)

#Step 2: standardize the x-variables 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)



#Step 3: do the KNN modeling 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train)

## Step 4: make predictions
y_pred = knn.predict(x_test_scaled)

## Step 5: Check f1 score
f1_score(y_test,y_pred)

knn = KNeighborsClassifier()
knn.fit(x_train_scaled,y_train)
y_pred =knn.predict(x_test_scaled)
f1 =f1_score(y_test, y_pred)
print('score is', f1_score)

########################CEAT A PIPELINE #########

from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', MinMaxScaler()), ('knn',KNeighborsClassifier())])
pipe.fit(x_train,y_train)


#Make predicitions on the x_test and get f1_score
y_pred_pipe =pipe.predict(x_test)
f1= f1_score(y_test,y_pred_pipe)
print('score is',f1)


#create aparameter grid
param_grid = {'knn__n_neighbors': range(1,50), 'knn__p':[1,2]   }

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe,param_grid, verbose = 3, scoring='f1')

grid.fit(x_train,y_train)

#best params
grid.best_params_

''' {'knn__n_neighbors': 12, 'knn__p': 1}'''

##OPtimziing params
pipe = Pipeline([('scaler', MinMaxScaler()),
                 ('knn',KNeighborsClassifier(n_neighbors = 13, p = 1))])
pipe.fit(x_train,y_train)


#Make predicitions on the x_test and get f1_score
y_pred_pipe =pipe.predict(x_test)
f1= f1_score(y_test,y_pred_pipe)
print('score is',f1)

























































