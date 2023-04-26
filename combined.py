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
lower_skin = np.array([0, 0, 0])                #original: 0, 20, 70  # these #'s sets a limit on how dark/light
                                                #skin color should be
upper_skin = np.array([255, 255, 255])          #50, 255, 255
                        #but these were more accurate tbh
                        # and detects all colors as skin color.

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

    # Wait for the user to press any key to take a picture
    key = cv2.waitKey(0)

    # Check if the user pressed the 'q' key to quit
    if key == ord('q'):
        break

    # Extract the sub-region of the image corresponding to the rectangle
    sub_region = frame[y1:y2, x1:x2]

    # Convert the sub-region to the HSV color space
    hsv_sub_region = cv2.cvtColor(sub_region, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only skin color regions
    mask = cv2.inRange(hsv_sub_region, lower_skin, upper_skin)

    # Compute the mean RGB values of the pixels in the skin color regions
    mean_color = cv2.mean(sub_region, mask=mask)[:3]

    # Convert the mean RGB values to integers
    bgr_mean = np.round(mean_color).astype(np.uint8)

    # Convert the BGR values to RGB values
    rgb_mean = cv2.cvtColor(np.uint8([[bgr_mean]]), cv2.COLOR_BGR2RGB)[0][0]

    # Convert the RGB values to a hexadecimal string
    hex_mean = '#' + ''.join([hex(int(c)).lstrip('0x').zfill(2) for c in rgb_mean])

    # Print the hexadecimal value of the sub-region
    print("Hex Mean: ", hex_mean)

    # Display the frame with the mean skin color as a text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, hex_mean, (x1, y1-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with the rectangle and the mean skin color text overlay
    cv2.imshow("Foundation Recommendor", frame)

#Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()

# find the closest cluster to the user's input
distances = [np.linalg.norm(hex_mean - kmeans.cluster_centers_[i]) for i in range(kmeans.n_clusters)]
closest_cluster = np.argmin(distances)

# recommend products from the closest cluster
recommended_products = [p["name"] for p in products if labels[list(X).index(p["color"])] == closest_cluster]
print("Recommended products:", recommended_products)