import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features for visualization (sepal length, sepal width)
y = iris.target  # Target labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree classifier
dtree = DecisionTreeClassifier(max_depth=3)  # Limiting the depth to avoid overfitting

# Train the Decision Tree
dtree.fit(X_train, y_train)

# Predict on the test set
y_pred = dtree.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dtree, feature_names=iris.feature_names[:2], class_names=iris.target_names, filled=True)
plt.title('Decision Tree Visualization')
plt.show()

# Function to visualize the decision boundaries
def plot_decision_boundary(X, y, model, title="Decision Tree Decision Boundary"):
    # Create a mesh grid based on the feature range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the class for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    
    # Scatter plot of the data points (train + test) with true labels
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=100, edgecolor='k')
    
    # Legend for the true labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=list(iris.target_names))  # Convert array to list

    plt.title(title)
    plt.xlabel(iris.feature_names[0])  # Sepal length
    plt.ylabel(iris.feature_names[1])  # Sepal width
    plt.show()

# Plot decision boundary for training data
plot_decision_boundary(X_train, y_train, dtree, title="Decision Tree - Training Data")

# Plot decision boundary for test data
plot_decision_boundary(X_test, y_test, dtree, title="Decision Tree - Test Data")