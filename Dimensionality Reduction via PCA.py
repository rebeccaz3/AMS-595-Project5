import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Load the data from the .csv file
data = pd.read_csv('data.csv').values 

# Compute the covariance matrix
covariance_matrix = np.cov(data, rowvar=False)

# Perform eigenvalue decomposition on the covariance matrix
eigenvalues, eigenvectors = eigh(covariance_matrix)

# Identify the principal components
# Sort the indices from largest to smallest eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]

# Based on sorted indices, indicate which variable corresponds to which principal component
if sorted_indices[0]==0: 
    print("Height corresponds to the first principal component")
else: 
    print("Weight corresponds to the first principal component")
    
    
if sorted_indices[1]==0: 
    print("Height corresponds to the second principal component")
else: 
    print("Weight corresponds to the second principal component")

# Reorder eigenvalues
eigenvalues = eigenvalues[sorted_indices]

# Reorder eigenvectors of corresponding eigenvalues
eigenvectors = eigenvectors[:, sorted_indices]

# The first eigenvector is the principal component, it captures the most variance
principal_component = eigenvectors[:, 0]

# Total variance in the dataset is the sum of eigenvalues
total_variance = np.sum(eigenvalues)

# Variance explained by each principal component
variance_explained = eigenvalues / total_variance

# Print the variance explained by each component
print("Variance explained:", variance_explained)
first_component = variance_explained[0]*100
second_component = variance_explained[1]*100
print(f"First component: {first_component:.4f}%")
print(f"Second component: {second_component:.4f}%")

# Reduce the dataset to 1D by projecting it onto the first principal component that captures the most variance
projected_data = data @ principal_component 

# Plot the original data and the 1D projection
plt.figure(figsize=(8, 6))

# First, plot the original 2D data as a scatterplot
plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.7, label='Original Data')

# Next, plot the 1D projection data
plt.scatter(projected_data, np.zeros_like(projected_data), color='red', alpha=0.7, label='1D Projection')

# Clean up the figure display
plt.title('Original Data and 1D Projection onto Principal Component')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()

# Save the plot as a PNG file
plt.savefig('PCA_projection.png')

# Display the plot in the command window
plt.show()

