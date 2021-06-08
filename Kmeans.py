#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 07:51:07 2021

@author: jadelrez
"""

import pandas as pd

df= pd.read_csv('circles.csv')



##Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df=scaler.fit_transform(df)

#Kmean with k=2
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(scaled_df)

df['labels']=km.labels_


import seaborn as sns 
sns.scatterplot(x='x1',y='x2',data= df,hue ='labels')


######################### Ward method dendogram######################
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
linked=  linkage(scaled_df, method = 'ward')
dendrogram(linked)
plt.show()


#####################  Single method ##################################
linked=  linkage(scaled_df, method = 'single')
dendrogram(linked)
plt.show() # Majority of the time is doesn't work


#implement hierchical clustering using two clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 2, linkage = 'single')
hc.fit(scaled_df)

#Plot is
df['labels']=hc.labels_

sns.scatterplot(x='x1',y='x2',data= df,hue ='labels')

##Single method works in this case 


########Can you try different approaches (ward, single, complete, average) on the future datasel from previous week
###########################

df= pd.read_csv('future.csv')

##Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df=scaler.fit_transform(df)

#Kmean with k=2
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(scaled_df)


######################### Ward method dendogram######################
# Ward,single, complete, average
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
linked=  linkage(scaled_df, method = 'complete')
dendrogram(linked)
plt.show()

######################## Principal component analysis ########################
######################## A method of dimensionality reduction ###############




































