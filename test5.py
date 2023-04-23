import cv2
import face_recognition

# Load the image
image = cv2.imread('image.jpg')

# Load the video stream
video_capture = cv2.VideoCapture(0)

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

# Find all the faces in the video stream
while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)

    # Draw a box around each face in the image
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the video stream with the detected faces
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
