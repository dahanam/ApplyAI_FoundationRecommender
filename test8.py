import cv2
import numpy as np

# Define the threshold range of RGB values for skin color
lower_skin = np.array([0, 48, 80], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Create a VideoCapture object to capture frames from the webcam
capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not capture.isOpened():
    print("Unable to open webcam")
    exit()

# Loop over frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = capture.read()

    # Check if the frame was read successfully
    if not ret:
        print("Unable to read frame from webcam")
        break

    # Get the center of the frame
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

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract the sub-region of the image corresponding to the rectangle
    sub_region = frame[y1:y2, x1:x2]

    # Convert the sub-region to the HSV color space
    hsv = cv2.cvtColor(sub_region, cv2.COLOR_BGR2HSV)

    # Apply a mask to the HSV image to extract only the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(sub_region, sub_region, mask=mask)

    # Compute the mean BGR values of the pixels in the skin sub-region
    bgr_mean = cv2.mean(skin)

    # Convert the BGR values to RGB values
    rgb_mean = cv2.cvtColor(np.uint8([[bgr_mean[:3]]]), cv2.COLOR_BGR2RGB)[0][0]

    # Print the RGB values of the skin sub-region
    print("Skin RGB: ", rgb_mean)

    # Display the frame
    cv2.imshow("Window Name", frame)

    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()
