##produces continuous rgb outputs
# to quit press q

import cv2
import numpy as np

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
    #print("RGB Mean: ", result_img)


    # Display the frame with the mean skin color as text on it
    cv2.rectangle(frame, (0, 0), (200, 40), (0, 0, 0), -1)
    cv2.putText(frame, "Mean Skin Color: ({}, {}, {})".format(rgb_mean[0], rgb_mean[1], rgb_mean[2]), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #cv2.putText(frame,output_img)

    result_img = np.zeros((200, 400, 3), dtype=np.uint8)
    if rgb_mean[0] > 160 and rgb_mean[1] > 120 and rgb_mean[2] > 100:
        cv2.putText(result_img, "Light skin tone", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(result_img, (50, 50), (150, 150), (255, 218, 185), -1)
    elif rgb_mean[0] > 130 and rgb_mean[1] > 80 and rgb_mean[2] > 70:
        cv2.putText(result_img, "Medium skin tone", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(result_img, (50, 50), (150, 150), (210, 180, 140), -1)
    else:
        cv2.putText(result_img, "Dark skin tone", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(result_img, (50, 50), (150, 150), (139, 69, 19), -1)


    #cv2.imshow("Window Name",result_img)
    cv2.imshow("Color Tone", frame)
    # concatenate the two images horizontally
    #output_img = cv2.hconcat([result_img, frame])

    # display the resulting image
    #cv2.imshow("Window Name", frame) #shows the frame with the rectangle
    #cv2.imshow("Window Name", output_img) # just takes a pic and outputs the color in console



    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close all windows
capture.release()
cv2.destroyAllWindows()



