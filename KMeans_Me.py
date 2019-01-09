# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:00:56 2018

@author: carto
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Elbow method to optimize number of clusters
from sklearn.cluster import KMeans
wCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 4)
    kmeans.fit(X)
    wCSS.append(kmeans.inertia_)
plt.plot(range(1,11), wCSS)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('wCSS')
plt.show()

#Applying KMeans to data
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 4)
y_kmeans=kmeans.fit_predict(X)

#Visualizing clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s=100, c='blue', label = 'C1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s=100, c='red', label = 'C2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s=100, c='green', label = 'C3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s=100, c='yellow', label = 'C4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s=100, c='purple', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='black', label = 'Centroid')
plt.title('Clusters of Clients')
plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()
plt.show()