# AMS-595-Project5
4 projects: the PageRank Algorithm, Dimensionality Reduction via PCA, Linear Regression via Least Squares, and Gradient Descent for Minimizing Loss Function.


# (1) PageRank Algorithm
This Python script implements a simplified version of the PageRank algorithm to rank web pages based on a defined network structure. It uses matrix operations to simulate the probability that a user clicks through links on different pages, ultimately determining the most likely destinations in the web network.

**How It Works**
- Define a matrix M, where each entry M[i, j] represents the probability that a user on page j will click a link to page i.
- Uses the eig function from scipy.linalg, the script calculates the eigenvalues and eigenvectors of matrix M. The dominant eigenvector, associated with the largest eigenvalue, represents the steady-state rank distribution across pages.
- The PageRank vector (v) is iteratively updated by multiplying it with the matrix M until convergence is reached, defined by a small tolerance threshold.
- Once convergence is achieved, the PageRank scores are normalized, and the page with the highest rank is identified.

**Key Parameters**
- tolerance: Controls the convergence threshold (default 1e-6).
- M: The transition matrix representing the link structure of the web pages.

**Output**
- The number of iterations taken to reach convergence.
- The final PageRank scores for each page in the network.
- The highest-ranked page.


# (2) PCA Dimensionality Reduction
This Python script uses Principal Component Analysis (PCA) to reduce the dimensionality of a dataset containing height and weight measurements for 100 individuals. The PCA method captures the most significant variance in the data while simplifying its structure.

**How It Works**
- Compute the covariance matrix to understand how height and weight vary together.
- Perform eigenvalue decomposition to identify the principal components that capture the most variance.
- Projects the 2D data onto the first principal component, effectively reducing the dataset to 1D while retaining key variance.

**Inputs** 
- File called data.csv which heights in column 1, and weights in column 2

**Output**
- The first principal component (corresponding to Weight) captures 55.74% of the variance.
- The second principal component (corresponding to Height) captures 44.26% of the variance.
- Visualization comparing the original 2D data and the 1D projection, saved as PCA_projection.png.


# Linear Regression via Least Squares
This Python script implements Linear Regression via the Least Squares method to predict house prices based on factors such as square footage, number of bedrooms, and age of the house. The model is designed for a real estate company aiming to estimate housing prices with readily available data.

**How It Works**
- Matrix X contains data about house features: square footage, number of bedrooms, and age.
- Vector array y contains data about corresponding house prices (in $1000s)
- Using lstsq from scipy.linalg, the script calculates the regression coefficients that minimize the residual sum of squares, providing weights for each house feature.
- Once trained, the model can be used to predict the price of a new house based on given features.

**Inputs**
- square footage
- number of bedrooms
- age of the house

**Output**
- The regression coefficients (weights) used in the model for square footage, bedrooms, and age. 
- A predicted price for a sample house with specified features, based on the model

**Test Example**
- The script includes a sample prediction for a house with: 2400 square feet, 3 bedrooms, and 20 years old
- Output shows that the predicted price of this house is $1448750.00
