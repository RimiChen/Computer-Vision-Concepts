import cv2

### blue, green, read
### -1, cv2.IMREAD_COLOR: color image, no transparency
###  0, cv2.IMREAD_GRAYSCALE: grayscale mode
###  1, cv2.IMREAD_UNCHANGED: including alpha channel
image = cv2.imread("assets/EIPR_2_4.png", 0)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("new_EIPR_2_4.png", image)

cv2.imshow("Image", image)
### 0 -> infinite, number -> number of seconds
cv2.waitKey(0)
cv2.destroyAllWindows()