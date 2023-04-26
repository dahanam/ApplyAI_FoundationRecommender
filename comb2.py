## hex capture from user is done
# clustering from the dataset is done

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import cv2


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
#plt.show()

# Create a VideoCapture object to capture frames from the webcam
capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not capture.isOpened():
    print("Unable to open webcam")
    exit()

# Get the center of the frame
_, frame = capture.read()
height, width, channels = frame.shape
center = (int(width/2), int(height/2))

# Set the desired width and height of the rectangle
rect_width = 100
rect_height = 50

# Calculate the coordinates of the top-left and bottom-right corners of the rectangle
x1 = int(center[0] - rect_width/2)
y1 = int(center[1] - rect_height/2)
x2 = int(center[0] + rect_width/2)
y2 = int(center[1] + rect_height/2)

# Define the lower and upper bounds of the skin color in RGB
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([50, 255, 255])

# Loop over frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = capture.read()

    # Check if the frame was read successfully
    if not ret:
        print("Unable to read frame from webcam")
        break

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with the rectangle
    cv2.imshow("Foundation Recommendor", frame)

    # Wait for the user to press any key to take a screenshot of the rectangle
    key = cv2.waitKey(1)
    if key == ord(' '):
        # Crop the region of interest from the frame
        roi = frame[y1:y2, x1:x2]

        # Convert the region of interest from BGR to RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Resize the region of interest to 1x1 pixel
        roi = cv2.resize(roi, (1, 1))

        # Get the RGB value of the pixel in the region of interest
        r, g, b = roi[0][0]

        # Get the closest cluster label to the RGB value
        closest_label = kmeans.predict([[r, g, b]])

        # Get the hex code of the closest cluster label
        closest_hex = makeup_df[np.squeeze(makeup_df['Cluster'] == closest_label)]['hex'].values[0]

        # Display the recommended foundation shade
        print("Recommended foundation shade:", closest_hex)

        # Wait for the user to press any key to exit
        cv2.waitKey(0)
        break
#Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()
