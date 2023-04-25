## rgb capture works
# k-means clustering seems to work but need to visualize it
# add the recommendor

## early version (dont use)

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

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

# Initialize the cluster centers
last_centers = None

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

    # Check if there are any skin color pixels in the sub-region
    if np.sum(mask) > 0:
        # Compute the cluster centers of the pixels in the skin color regions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        num_clusters = 1
        flags = cv2.KMEANS_RANDOM_CENTERS
        data = sub_region[mask > 0].reshape(-1, 3).astype(np.float32)
        ret, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, 10, flags)

        # Store the cluster centers for the current sub-region
        last_centers = centers

        # Convert the BGR values to RGB values
        # Convert the BGR values to RGB values
        rgb_centers = cv2.cvtColor(centers.astype(np.uint8), cv2.COLOR_BGR2RGB)


