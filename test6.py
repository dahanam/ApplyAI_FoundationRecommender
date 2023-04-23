import cv
import cv2_rectangle_around_center
import cv2_rgb
import cv2_stack_images
import cv2

path = r"C:\Users\dahan\OneDrive\Documents\image1.jpg"
img = cv2.imread(path)
center = (120, 50)
radius = 20
color = (255, 0, 0)
thickness = 2

cv2.circle(img, center, radius, color, thickness)

cv2.imshow("Image", img)


cv2.waitKey(0)

cv2.destroyAllWindows()
