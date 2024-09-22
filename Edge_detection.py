import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


folder = "Obj_detection_small/"

# Load images from a folder and resize them
def load_images_from_folder(folder):
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize the image to a consistent size
            # img_resized = cv2.resize(img, IMAGE_SIZE)
            # images.append(img_resized)
            filenames.append(filename)
    return filenames

filenames = load_images_from_folder(folder)
print(filenames)


# Load the image in grayscale
image = cv2.imread(folder + filenames[0], cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction

# Compute the magnitude of the gradient
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.show()

####-----------------------------------

# Define Prewitt kernels
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# Apply Prewitt edge detection using cv2.filter2D()
prewitt_x_edges = cv2.filter2D(image, -1, prewitt_x)
prewitt_y_edges = cv2.filter2D(image, -1, prewitt_y)

# Compute the gradient magnitude
prewitt_magnitude = np.sqrt(prewitt_x_edges**2 + prewitt_y_edges**2)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Prewitt Magnitude')
plt.imshow(prewitt_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.show()


### -----------------------

# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.show()