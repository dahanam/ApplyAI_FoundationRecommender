## just testing the print

## FINAL FINAL VERSION <3

import cv2
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import colorsys

# Load the Makeup Shades Dataset
makeup_data = pd.read_csv("C:/Users/dahan/OneDrive/Documents/shades.csv")

# Extract RGB values of pixels from the dataset
makeup_rgb = makeup_data[['hex']].apply(lambda x: pd.Series([int(x['hex'][i:i+2], 16) for i in (0, 2, 4)]), axis=1)

# Define the number of clusters (k)
k = 6

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
    mask = cv2.inRange(hsv_sub_region, lower_skin, upper_skin)

    # Compute the mean RGB values of the pixels in the skin color regions
    mean_color = cv2.mean(sub_region, mask=mask)[:3]

    # Convert the mean RGB values to integers
    bgr_mean = np.round(mean_color).astype(np.uint8)

    # Convert the BGR values to RGB values
    rgb_mean = cv2.cvtColor(np.uint8([[bgr_mean]]), cv2.COLOR_BGR2RGB)[0][0]

    # Convert the RGB values to a hexadecimal integer
    hex_mean = '#{0:02x}{1:02x}{2:02x}'.format(rgb_mean[0], rgb_mean[1], rgb_mean[2])

    #Find the index of the closest cluster center to the mean color
    distances = [np.sqrt(np.sum((kmeans.cluster_centers_[i] - bgr_mean) ** 2)) for i in range(k)]
    closest_cluster_index = np.argmin(distances)

    #Get the makeup shade corresponding to the closest cluster center
    makeup_shade = makeup_data.loc[kmeans.labels_ == closest_cluster_index]['hex'].values[0]

    #printing
    index = makeup_data.index[makeup_data['hex'] == makeup_shade].tolist()[0]
    recommended_shade = makeup_data.iloc[index]
    print(f"Recommended makeup shade: {recommended_shade['brand']} - {recommended_shade['product']} - {recommended_shade['hex']}")

##Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()