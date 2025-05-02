import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# 1. Choose a classification dataset and normalize features.

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
# Using a test size of 20% as an example
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Use KNeighbors Classifier from sklearn.
# 3. Experiment with different values of K.

# Experiment with K from 1 to 20
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    # Initialize the KNN classifier with the current K
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Plot the accuracy for different K values to find the optimal K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Find the optimal K (the one with the highest accuracy)
optimal_k_index = np.argmax(accuracy_scores)
optimal_k = k_values[optimal_k_index]
print(f"Optimal K value: {optimal_k}")
print(f"Highest accuracy with optimal K: {accuracy_scores[optimal_k_index]}")

# Train the KNN model with the optimal K
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)
y_pred_optimal = knn_optimal.predict(X_test_scaled)


# 4. Evaluate model using accuracy, confusion matrix.

# Evaluate the model with the optimal K
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
conf_matrix_optimal = confusion_matrix(y_test, y_pred_optimal)

print(f"\nModel Evaluation with Optimal K ({optimal_k}):")
print(f"Accuracy: {accuracy_optimal}")
print("Confusion Matrix:")
print(conf_matrix_optimal)

# 5. Visualize decision boundaries.

# This visualization is typically for 2D data.
# For the Iris dataset with 4 features, we'll visualize using the first two features as an example.
# You can change the feature indices to visualize different pairs of features.

feature_index_1 = 0  # Index of the first feature (e.g., Sepal Length)
feature_index_2 = 1  # Index of the second feature (e.g., Sepal Width)

X_vis = X_train_scaled[:, [feature_index_1, feature_index_2]]
y_vis = y_train

# Create a meshgrid of points to plot the decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train a KNN classifier on the 2D data for visualization
knn_vis = KNeighborsClassifier(n_neighbors=optimal_k) # Use the optimal K
knn_vis.fit(X_vis, y_vis)

# Predict the class for each point in the meshgrid
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=.8)

# Plot the training points
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=plt.cm.RdYlBu, edgecolor='k', s=20)

# Add labels and title
plt.xlabel(iris.feature_names[feature_index_1])
plt.ylabel(iris.feature_names[feature_index_2])
plt.title(f'KNN Decision Boundaries (k={optimal_k})')

# Add a legend for the classes
legend = plt.legend(*scatter.legend_elements(), title="Classes")
plt.add_artist(legend)

plt.show()
