import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load dataset (example: positive and negative image samples)
# Load images from a folder
# Resize images to a consistent size
# Resize images to a consistent size
IMAGE_SIZE = (128, 128)  # Define a fixed size for all images

# Load images from a folder and resize them
def load_images_from_folder(folder, target_string):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize the image to a consistent size
            img_resized = cv2.resize(img, IMAGE_SIZE)
            images.append(img_resized)
            filenames.append(filename)
            # Example: label assignment logic (depends on your dataset)
            if target_string in filename:
                labels.append(1)
            else:
                labels.append(0)
    return images, labels, filenames

# Example folder path (adjust this as necessary)
folder_path = "Obj_detection_small/"
target_string = "apple"
X, y, filenames = load_images_from_folder(folder_path, target_string)

# Feature extraction using HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(feature)
    return np.array(hog_features)

# Extract HOG features
X_hog = extract_hog_features(X)

# Split the data into training and testing sets (we also split filenames for reference)
X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(
    X_hog, y, filenames, test_size=0.2, random_state=42)

# Show which images are in the train and test sets
print("Training Images:", train_filenames)
print("Test Images:", test_filenames)

# Train a Linear SVM
svm = LinearSVC()
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the SVM results using PCA to reduce HOG features to 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the decision boundary and the points in 2D space
plt.figure(figsize=(10, 6))

# Plot the training data
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', label='Train Data')

# Plot the test data
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test Data')

plt.title("PCA Projection of HOG Features and SVM Results")
plt.xlabel("Negtive")
plt.ylabel("Positive")
plt.legend()
plt.show()

# Overlay the prediction results on the test images and show them
for i, filename in enumerate(test_filenames):
    img = cv2.imread(os.path.join(folder_path, filename))
    img_resized = cv2.resize(img, IMAGE_SIZE)
    
    # Add prediction result on the image
    label = 'Positive' if y_pred[i] == 1 else 'Negative'
    ground_truth = 'Positive' if y_test[i] == 1 else 'Negative'
    
    cv2.putText(img_resized, f'Pred: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_resized, f'GT: {ground_truth}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show the image
    cv2.imshow(f'Result: {filename}', img_resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()