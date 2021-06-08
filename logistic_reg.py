import pandas as pd
df1 = pd.read_csv("cancer-2.csv")
df2 = pd.read_csv("default-1.csv")

import seaborn as sns
import matplotlib.pyplot as plt

#Exploring relationships visually 
sns.boxplot(x= "default", y= "balance", data = df2)

#Dumify the data 
df = pd.get_dummies(df2, drop_first = True)

##### First do a linear regression ####

##Separate X and Y 
x =  df[["balance","student_Yes","income"]]
y= df[["default_Yes"]]

##Train-Test split 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train,y_test= train_test_split(x,y,test_size = 0.3, random_state = 101)

#from sklearn.linear_model import LinearRegression 
#lm = LinearRegression()
#lm.fit(x_train,y_train)

#y_pred_lm = lm.predict(x_test)
# if probability is more that 0.5 then user will default 

from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression(solver= 'liblinear')
logmodel.fit(x_train,y_train)

##Make predictions
y_pred = logmodel.predict(x_test)
## To get the specific probability 
y_probab =logmodel.predict_proba(x_test)


##To evalkuate the coefficients , find b0 and b1
logmodel.coef_
logmodel.intercept_
#Graph the long model
bing = pd.DataFrame(y_probab)
plt.scatter(x_test,bing[1])

#Build confusion matrix and get predictive values of the log model 
from sklearn.metrics import confusion_matrix, f1_score
C_mat = pd.DataFrame(confusion_matrix(y_test,y_pred, labels=[0,1]), 
                     index = ["Actual:0","Actual:1"], columns=["Pred:0","Pred:1"])
print(C_mat)
f1_score(y_test,y_pred)














