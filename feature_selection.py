#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:13:55 2021

@author: jadelrez
"""

import pandas as pd
df=pd.read_csv("dummy_data.csv")

#Run regression to predict the sallary of profs
df_dummy = pd.get_dummies(df,drop_first= True)

#Separate into x and y variables
x= df_dummy.drop('sl', axis = 1)
y = df_dummy['sl']

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y,test_size= 0.3, random_state =1)

#build the model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)



y_pred = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
print("R square is", r2_score(y_test,y_pred))
#TEST METRICS on Testing
mse=mean_squared_error(y_test,y_pred)
rmse=mse**0.5

print("rmse is", rmse)

#Evaluation model on training data
y_pred_train = model.predict(x_train)
from sklearn.metrics import r2_score,mean_squared_error
print("R square is", r2_score(y_train,y_pred_train))
#TEST METRICS
mse=mean_squared_error(y_train,y_pred_train)
rmse=mse**0.5

#TEST METRICS on Training
print("rmse is", rmse)



###Titanic Data 
df= pd.read_csv('titanic_data-5.csv')
df.info()
#Check for missing values
import seaborn as sns 
sns.heatmap(df.isnull())
#Drop the rows with missing values
df1=df.dropna()

#Option 2:instead of dropping rows, Drop columns with missing values
df2= df.drop(['Cabin','Embarked','Age'], axis = 1)

#Instead of dropping. Replace the missing values 
sns.boxplot(data=df['Age'])
median_age = df['Age'].median()
df['Age']=df['Age'].fillna(median_age)

#Embarked column
df['Embarked'].value_counts()
df['Embarked']=df['Embarked'].fillna('S')

#Cabin Column
df['Cabin'].value_counts()

#### AMES HOUSING 
df= pd.read_csv('Ameshousing.csv')
df.info()
#DROP unwanted columns
df= df.drop(['Order','PID'],axis = 1)
#DROP rows with missing values
df=df.dropna()

#Regression to predict the salesprice of a house 
#Separate the x and y variables
x=df.drop('SalePrice',axis = 1)
y=df['SalePrice']
#Train test spliot
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


from sklearn.linear_model import LinearRegression #import regression model
lm= LinearRegression()
lm.fit(x_train,y_train)
predictions = lm.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print("r squared is",r2_score(y_test, predictions)) # The r-square error
mse=mean_squared_error(y_test,predictions)
rmse=mse**0.5
print("rmse is",rmse)
#These are the results without feature selection
'''r squared is 0.8451202107848694
rmse is 31796.36504263581'''

#Can we improve the model by doing feature selection 
#feature slection using sklearn to see if it can be improve


from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score,mean_squared_error

for i in range(1,37):
    bestfeatures = SelectKBest(score_func=f_regression, k = i)
    new_x = bestfeatures.fit_transform(x,y)
    x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.3,random_state=1)
    lm= LinearRegression()
    lm.fit(x_train,y_train)
    predictions = lm.predict(x_test)
    mse=mean_squared_error(y_test,predictions)
    rmse=mse**0.5
    print("rmse is",rmse)
print('rmse is minimized when k = 26')
  
    


   





















































