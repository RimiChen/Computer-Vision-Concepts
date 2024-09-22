import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
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

# Load the blurred image
image = img_as_float(io.imread(folder + filenames[0], as_gray=True))

# Initial estimates
latent_est = image.copy()  # The estimate of the deblurred image (same size as image)
psf_size = 15  # Adjust based on expected blur
psf_est = np.ones((psf_size, psf_size)) / (psf_size ** 2)  # Initial PSF estimate

# Parameters
num_iter = 20  # Number of iterations
epsilon = 1e-8  # Small constant to avoid division by zero

for i in range(num_iter):
    # Convolve latent image with PSF estimate
    est_blur = cv2.filter2D(latent_est, -1, psf_est, borderType=cv2.BORDER_REFLECT)
    
    # Compute ratio between observed image and estimated blur
    relative_blur = image / (est_blur + epsilon)
    
    # Update latent image estimate
    latent_est *= cv2.filter2D(relative_blur, -1, psf_est[::-1, ::-1], borderType=cv2.BORDER_REFLECT)
    
    # Normalize latent image
    latent_est = np.clip(latent_est, 0, 1)
    
    # Update PSF estimate
    # Calculate the new PSF by using the current estimate of the latent image (use only the region of the PSF size)
    region_of_interest = latent_est[:psf_size, :psf_size]
    
    # Update the PSF based on the relative blur and the region of interest
    psf_est = cv2.filter2D(relative_blur, -1, region_of_interest[::-1, ::-1], borderType=cv2.BORDER_REFLECT)
    
    # Normalize PSF
    psf_est /= psf_est.sum()

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Blurred Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Estimated PSF')
plt.imshow(psf_est, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Deblurred Image')
plt.imshow(latent_est, cmap='gray')
plt.axis('off')

plt.show()