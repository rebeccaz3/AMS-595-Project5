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


# (3) Linear Regression via Least Squares
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

# (4) Gradient Descent for Loss Function Minimization
This Python script demonstrates the use of gradient descent to minimize a loss function by optimizing a matrix X to approximate a target matrix A. The script uses the scipy.optimize.minimize function with the BFGS method, a gradient-based optimization technique, to minimize the difference between the current guess (X) and the target matrix (A). The goal is to find the matrix X that minimizes the squared difference between itself and A.

**How It Works**
- Define a loss function that calculates the squared difference between the current matrix X and the target matrix A. The objective is to minimize this loss, making X as close as possible to A.
- Gradient Descent optimization process is used. At each step, the script computes the gradient (the derivative of the loss function) and updates the matrix X in the direction that reduces the loss. This process is repeated until the loss becomes sufficiently small or the maximum number of iterations is reached.
- BFGS Algorithm: The script uses the BFGS method (a gradient-based optimization algorithm) from scipy.optimize.minimize. This method is efficient for large problems, as it approximates the inverse Hessian matrix to guide the search for the optimal solution.
- The script tracks how the loss function evolves during the optimization and visualizes this progression with a plot showing the loss values over iterations. 

**Key Parameters**
- Matrix A: This is the target matrix you want X to approximate. It is randomly generated in the example code as a 100x50 matrix. You can adjust the size of A by modifying the dimensions in np.random.rand(m, n) where m and n are the number of rows and columns, respectively.
- Matrix X_init: Initial guess for X, the matrix that will be optimized. It has the same shape as A and is initialized randomly as well. You can modify the initialization of X_init to experiment with different starting points.
- Threshold (threshold=1e-6): This parameter controls the stopping criterion for the optimization. When the change in the loss function is less than this threshold, the optimization stops. You can adjust this value to make the optimization more or less sensitive.
- Maximum Iterations (max_iter=1000): This parameter specifies the maximum number of iterations for the gradient descent optimization. If the algorithm doesn't converge before this limit, it will stop.
- You can change this value to allow for more or fewer iterations, depending on how fast or slow you want the optimization process to run.
  
**Output**
- Statement indicating whether the optimization was successful.
- Current function value (should be small, nearly 0)
- Number of iterations
- Number of function evaluations
- Number of gradient evaluations 
- Optimized Matrix X: This matrix has been adjusted to minimize the loss function and should closely resemble the target matrix A.
- Visualization of Loss Reduction Over Iterations: A plot shows the loss values (y-axis) versus the iteration number (x-axis), allowing you to see how the loss decreases over time. The plot is saved as Loss_values.png and displayed in the command window.
