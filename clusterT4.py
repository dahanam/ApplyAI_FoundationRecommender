## maybe

import cv2
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import webcolors

# Load the Makeup Shades Dataset
makeup_data = pd.read_csv("C:/Users/dahan/OneDrive/Documents/shades.csv")

# Extract RGB values of pixels from the dataset
makeup_rgb = makeup_data[['hex']].values

# Convert hex color codes to RGB values
makeup_rgb = [webcolors.hex_to_rgb(hex_color[0].strip()) for hex_color in makeup_rgb]


# Define the number of clusters (k)
k = 5

# Perform k-means clustering on the RGB values
kmeans = KMeans(n_clusters=k, random_state=0).fit(makeup_rgb)

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
    cv2.imshow("Foundation Recommender", frame)

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
    mask = cv2.inRange(hsv_sub_region,lower_skin, upper_skin)

    # Apply a Gaussian blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Perform k-means clustering on the colors in the sub-region of the image
    colors = cv2.cvtColor(sub_region, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    labels = kmeans.predict(colors)

    # Determine the most common color in the sub-region of the image
    counts = np.bincount(labels)
    most_common_color = kmeans.cluster_centers_[np.argmax(counts)]

    # Convert the most common color to its corresponding shade name
    closest_colors = webcolors.rgb_to_name(most_common_color)

    # Print the name of the closest shade
    print("The closest shade is:", closest_colors)

# Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()

