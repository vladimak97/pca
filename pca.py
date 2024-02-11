
# Implement Principal Component Analysis (PCA) in Python for dimensionality reduction.

import numpy as np

def pca(X, num_components):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    standardized_data = (X - mean) / std_dev
    cov_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :num_components]
    projected_data = np.dot(standardized_data, top_eigenvectors)
    return projected_data

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
num_components = 2
projected_data = pca(X, num_components)
print(f"Projected Data: {projected_data}")
