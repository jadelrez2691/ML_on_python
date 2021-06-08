#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 08:51:42 2021

@author: jadelrez
"""


#################################### KNN CLUSTERING ########################
import pandas as pd
df= pd.read_csv('future.csv')

import seaborn as sns
sns.pairplot(df)

#Since normal curve is skwed  will scale the variabels using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df =scaler.fit_transform(df)

#random guess k=4
from sklearn.cluster import KMeans
km4 = KMeans(n_clusters=4 , random_state = 0)
km4.fit(scaled_df)#clusters are found in this step 

km4.labels_

#Visualiize the clusters 
df['label']= km4.labels_

#Plot it
sns.scatterplot(x ='INCOME', y = 'SPEND', data =df, hue = 'label',
                palette = 'Set1')

##Using methods to finding k
from sklearn.metrics import silhouette_score
wcv=[]
silk_score= []

for i in range(2,11):
    km=KMeans(n_clusters= i, random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))
    
import matplotlib.pyplot as plt 
plt.plot(range(2,11), wcv    )
plt.xlabel('no of clusters')
plt.ylabel('within cluster variation')
plt.grid()

plt.plot(range(2,11), silk_score   )
plt.xlabel('no of clusters')
plt.ylabel('silk score')
plt.grid()



##Inclass Kmeans
## do knn for k = 3 and interpret your results 

from sklearn.cluster import KMeans
km3 = KMeans(n_clusters=3 , random_state = 0)
km3.fit(scaled_df)#clusters are found in this step 

km3.labels_

df['label']= km3.labels_

sns.scatterplot(x ='INCOME', y = 'SPEND', data =df, hue = 'label',
                palette = 'Set1')

# 0: high income, high spending 
# 1: low income, high spending
# 2: low income, low spending


########### FOOD CSV #########################


import pandas as pd
df_= pd.read_csv('food.csv')

df = df_.drop('Item', axis=1)

import seaborn as sns
sns.pairplot(df)

#Since normal curve is skwed  will scale the variabels using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df =scaler.fit_transform(df)

##Using methods to finding k
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
wcv=[]
silk_score= []

for i in range(2,11):
    km=KMeans(n_clusters= i, random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))
    
import matplotlib.pyplot as plt 
plt.plot(range(2,11), wcv    )
plt.xlabel('no of clusters')
plt.ylabel('within cluster variation')
plt.grid()

plt.plot(range(2,11), silk_score   )
plt.xlabel('no of clusters')
plt.ylabel('silk score')
plt.grid()


km4 = KMeans(n_clusters=3 , random_state = 0)
km4.fit(scaled_df)#clusters are found in this step 

km4.labels_

#Visualiize the clusters 
df_['label']= km4.labels_


## Interpret the results
#cluster 0 
df_.loc[df_['label']==0].describe()
#high calories and fatty foods


#cluster 1
df_.loc[df_['label']==1].describe()
#medium calorie and high calcium

#cluster 2
df_.loc[df_['label']==2].describe()
#healthy food with high iron

################################# HIERCHICAL CLUSTERING #####################


































