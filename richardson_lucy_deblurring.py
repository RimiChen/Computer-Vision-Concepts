import cv2
from skimage import restoration, io
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


# Load the blurred image (assumed to be grayscale or convert it to grayscale)
image = io.imread(folder + filenames[0], as_gray=True)

# Initial estimate of the PSF
psf_size = 15  # Adjust based on expected blur
psf = np.ones((psf_size, psf_size)) / (psf_size ** 2)

# Perform Richardson-Lucy deconvolution
num_iter = 30  # Number of iterations
deconvolved_image = restoration.richardson_lucy(image, psf, num_iter=num_iter)

# Display the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Original Blurred Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Deblurred Image (Richardson-Lucy)')
plt.imshow(deconvolved_image, cmap='gray')
plt.axis('off')

plt.show()