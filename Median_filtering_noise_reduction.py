import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "Blur_small/"

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

# Load a noisy image 
image = cv2.imread(folder + filenames[0], cv2.IMREAD_GRAYSCALE)

# Apply Median filtering with a kernel size of 5
median_filtered = cv2.medianBlur(image, 5)

# Show the original and filtered images
plt.figure(figsize=(10, 5))

# Display the original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Display the Median filtered image
plt.subplot(1, 2, 2)
plt.title('Median Filtered Image (5x5 Kernel)')
plt.imshow(median_filtered, cmap='gray')
plt.axis('off')

plt.show()