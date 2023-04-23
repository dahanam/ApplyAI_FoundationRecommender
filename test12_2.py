## ends but now it only takes one pic and doesnt allow for
# user to align themselves with the square
# task to do: make sure that it takes a pic and is able to continue to the next part

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

    # Wait for 7 seconds to give the user time to align themselves to the square
    start_time = time.time()
    while (time.time() - start_time) < 7:
        # Check if the user pressed the 'q' key to quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Check if the 'q' key was pressed
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

    # Print the RGB values of the sub-region
    print("RGB Mean: ", rgb_mean)

    # Display the frame with the mean skin color as text on it
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "RGB Mean: " + str(rgb_mean), (10, 50), font, 1, (0, 255, 0), 2)

    #Show the frame with the text
    cv2.imshow("Foundation Recommendor", frame)

    # Check if the user pressed the 'q' key to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()