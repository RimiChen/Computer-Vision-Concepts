import cv2
import numpy as np
import matplotlib.pyplot as plt

image_1 = "2views/Computer/view1.png"
image_2 = "2views/Computer/view5.png"


# Load left and right images (rectified stereo pair)
left_image = cv2.imread(image_1, cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(image_2, cv2.IMREAD_GRAYSCALE)

# Create a stereo block matcher object
# You can use either StereoBM or StereoSGBM
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_image, right_image)

# Normalize the disparity map for display
disparity_normalized = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Display the disparity map
plt.figure(figsize=(10, 5))
plt.title('Disparity Map')
plt.imshow(disparity_normalized, cmap='gray')
plt.axis('off')
plt.show()

# Depth estimation (optional)
# Assuming you know the baseline (B), focal length (f), and disparity (d)
f = 0.8  # Focal length in some units (this should match your camera settings)
B = 0.2  # Baseline in the same units
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Calculate depth from disparity (ignoring divide-by-zero warnings)
with np.errstate(divide='ignore'):
    depth_map = (f * B) / (disparity + 1e-6)  # Avoid division by zero with a small constant

# Display the depth map
plt.figure(figsize=(10, 5))
plt.title('Depth Map')
plt.imshow(depth_map, cmap='jet')
plt.axis('off')
plt.show()