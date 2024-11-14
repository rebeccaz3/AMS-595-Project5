
import numpy as np
from scipy.linalg import eig

# Define the matrix M, representing the web network 
# Each entry M[i, j] is the probability that a user on page j will click a link to page i
M = np.array([
    [0, 0, 1/2, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 1/2, 0, 1/2],
    [1/3, 1/2, 1/2, 0]
])

# Get eigenvalues and eigenvectors of matrix M
eigenvalues, eigenvectors = eig(M)

# Find the index of the largest eigenvalue
dominant_index = np.argmax(eigenvalues.real)

# Get the eigenvector corresponding to the largest eigenvalue (the dominant eigenvector)
dominant_eigenvector = eigenvectors[:, dominant_index].real

# Start with an initial rank vector v (of ones) so all pages have equal proability
v = np.ones(4)

# Set a small tolerance value, used to check if we are close to convergence
tolerance = 1e-6

# Initialize difference to enter the loop
diff = 1

# Initialize an iteration counter
iterations = 0

# Multiply rank vector v by matrix M over and over again
# We will repeat the calculation until the difference between consecutive v's is small
while diff > tolerance:
    # Multiply M by the current rank vector v to get the next rank vector
    v_next = M @ v

    # Calculate the difference between v_next and v
    diff = np.sum(np.abs(v_next - v))  # sum of absolute differences

    # Update v to be the new rank vector (v_next)
    v = v_next

    # Increase the iteration count
    iterations += 1

# Display number of iterations 
print("Number of iterations until convergence: ", iterations)

# Normalize the resulting eigenvector
page_ranks = dominant_eigenvector / np.sum(dominant_eigenvector)
print("Final PageRank scores:", page_ranks)

# Find the page with the highest hank
# Get the index of the page with the highest PageRank score
highest_ranked_page = np.argmax(page_ranks)

# Display the highest ranked page
print("Page ranked the highest:", highest_ranked_page+1) # +1 because index starts at 0, but page starts at 1


