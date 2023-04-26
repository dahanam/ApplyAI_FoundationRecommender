#################### implementing k-means clustering

import numpy as np
from sklearn.cluster import KMeans
import sklearn.neighbors
import pandas as pd

## Load the dataset
data = pd.read_csv("C:/Users/dahan/OneDrive/Documents/shades.csv", delimiter=',')
#data = pd.read_csv(r"C:\Users\dahan\OneDrive\Documents\shades.csv", delimiter=',')

# Preprocess data by scaling
data = (data - data.mean(axis=0)) / data.std(axis=0)

# Choose number of clusters
k = 3

# Initialize K-means object
kmeans = KMeans(n_clusters=k, init='random')

# Fit K-means to data
kmeans.fit(data)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Evaluate quality of clusters using sum of squared distances
sse = kmeans.inertia_

import matplotlib.pyplot as plt
# Create a scatter plot of the data points colored by their cluster label
plt.scatter(data[:, 0], data[:, 1], c=labels)

# Create a scatter plot of the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, linewidths=3, color='r')

# Add axis labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering (k={})'.format(k))

# Show the plot
plt.show()