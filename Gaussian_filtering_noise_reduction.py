import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "Blur_small/"

# Load images from a folder and resize them
def load_images_from_folder(folder):
    images = []
    labels = []
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

# Load a noisy image (replace with your image path)
image = cv2.imread(folder + filenames[0], cv2.IMREAD_GRAYSCALE)

# Apply Gaussian filtering with a kernel size of 5x5 and sigmaX = 1
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

# Show the original and filtered images
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Display the Gaussian filtered image
plt.subplot(1, 2, 2)
plt.title('Gaussian Filtered Image')
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')

plt.show()