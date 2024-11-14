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
