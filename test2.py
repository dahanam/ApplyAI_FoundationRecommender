import cv2
import cv2_rectangle_around_center
import cv2_rgb
import cv2_stack_images


# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)


# Define the codec and create a VideoWriter object to save the video to disk
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Loop over frames from the camera until the user presses the 'q' key
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If we successfully read a frame, display it on the screen and write it to disk
    if ret:
        cv2.imshow('frame', frame)
        out.write(frame)

    # Wait for the user to press a key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
out.release()
cv2.destroyAllWindows()





#path = r"C:\Users\dahan\OneDrive\Documents\image1.jpg"
#img = cv2.imread(path)


center = (120, 50)
radius = 20
color = (255, 0, 0)
thickness = 2

cv2.circle(img, center, radius, color, thickness)

cv2.imshow("Image", img)


cv2.waitKey(0)

cv2.destroyAllWindows()
