import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal import wiener
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



# Simulate motion blur function (PSF - Point Spread Function)
def motion_blur_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    
    # Creating the motion blur PSF in the desired angle
    for i in range(length):
        psf[center, i] = 1
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
    psf = cv2.warpAffine(psf, rotation_matrix, (length, length))
    psf /= psf.sum()  # Normalize the PSF
    return psf

# Apply motion blur to the image
def apply_motion_blur(image, psf):
    return convolve2d(image, psf, 'same')

# Wiener Deconvolution function
def wiener_deconvolution(image, psf, K):
    psf_pad = np.pad(psf, [(0, image.shape[0] - psf.shape[0]), (0, image.shape[1] - psf.shape[1])], 'constant')
    psf_fft = np.fft.fft2(psf_pad)
    image_fft = np.fft.fft2(image)
    
    psf_fft_conj = np.conj(psf_fft)
    
    wiener_filter = psf_fft_conj / (psf_fft * psf_fft_conj + K)
    deblurred_fft = image_fft * wiener_filter
    
    deblurred = np.fft.ifft2(deblurred_fft)
    return np.abs(deblurred)

# Load and prepare the image (grayscale)
image = cv2.imread(folder + filenames[0], cv2.IMREAD_GRAYSCALE)

# Simulate a motion blur (e.g., length 15, angle 30 degrees)
psf = motion_blur_psf(length=15, angle=30)

# Apply the motion blur to the image
blurred_image = apply_motion_blur(image, psf)

# Apply Wiener deconvolution to reverse the blur
K = 0.01  # A small constant to handle noise
deblurred_image = wiener_deconvolution(blurred_image, psf, K)

# Plot the results
plt.figure(figsize=(15, 5))

# Display the original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Display the blurred image
plt.subplot(1, 3, 2)
plt.title('Motion Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

# Display the deblurred image
plt.subplot(1, 3, 3)
plt.title('Deblurred Image (Wiener)')
plt.imshow(deblurred_image, cmap='gray')
plt.axis('off')

plt.show()