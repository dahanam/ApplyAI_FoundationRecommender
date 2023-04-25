import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
makeup_df = pd.read_csv("C:/Users/dahan/OneDrive/Documents/shades.csv")

# Extract the hex color codes from the dataset
hex_codes = makeup_df['hex'].values

# Convert the hex color codes to RGB values
rgb_values = np.array([list(int(hex_code.strip('#')[i:i+2], 16) for i in (0, 2, 4)) for hex_code in hex_codes])

# Define the number of clusters
num_clusters = 5

# Initialize the k-means clustering algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the algorithm to the data
kmeans.fit(rgb_values)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the dataset
makeup_df['Cluster'] = labels

# Visualize the clusters
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2], c=labels.astype(float))
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.show()
