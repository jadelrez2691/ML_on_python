#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:17:32 2021

@author: jadelrez
"""

#Import Auto dataframe 
import pandas as pd
df2=pd.read_csv("shark_attacks-1.csv")
import seaborn as sns
sns.pairplot(df2)


###MODEL 1
b0=df2[['IceCreamSales']]
yhat=df2['SharkAttacks']

#Split data into train and test
from sklearn.model_selection import train_test_split
b0_train,b0_test,yhat_train,yhat_test=train_test_split(b0,yhat,test_size=0.3,random_state=1
                                                       )

#We will now do the regression 
from sklearn.linear_model import LinearRegression
lm=LinearRegression() #Initialize the model

lm.fit(b0_train,yhat_train) #Training the model

#Summary of Coeff

predictions=lm.predict(b0_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(yhat_test,predictions))

mse=mean_squared_error(yhat_test, predictions)
print('rme is:',mse**0.5)

print(lm.coef_)

'R2 is: -0.007154544483181757 and rme is: 7.693580942486933'
'lm.coeff is: 0.4188564 '

###MODEL 2
b0=df2[['IceCreamSales','Temperature']]
yhat=df2['SharkAttacks']

#Split data into train and test
from sklearn.model_selection import train_test_split
b0_train,b0_test,yhat_train,yhat_test=train_test_split(b0,yhat,test_size=0.3,random_state=1)

#We will now do the regression 
from sklearn.linear_model import LinearRegression
lm=LinearRegression() #Initialize the model

lm.fit(b0_train,yhat_train) #Training the model

#Summary of Coeff

predictions=lm.predict(b0_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(yhat_test,predictions))

mse=mean_squared_error(yhat_test, predictions)
print('rme is:',mse**0.5)

print(lm.coef_)

'R2 is: 0.3091833833733332 rme is: 6.371795906949918'
'lm.coeff for IceCreamSales is 0.17946881 and lm.coeff for Temperature is 1.29905347'

##MODEL 3
b0=df2[['IceCreamSales']]
yhat=df2['Temperature']

#Split data into train and test
from sklearn.model_selection import train_test_split
b0_train,b0_test,yhat_train,yhat_test=train_test_split(b0,yhat,test_size=0.3,random_state=1)

#We will now do the regression 
from sklearn.linear_model import LinearRegression
lm=LinearRegression() #Initialize the model

lm.fit(b0_train,yhat_train) #Training the model

#Summary of Coeff

predictions=lm.predict(b0_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(yhat_test,predictions))

mse=mean_squared_error(yhat_test, predictions)
print('rme is:',mse**0.5)

print(lm.coef_)

'R2 is: 0.36855985510347733 rme is: 3.3228193007791598'
'lm.coeff is: 0.18427847'





