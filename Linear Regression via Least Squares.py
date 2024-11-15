import numpy as np
from scipy.linalg import lstsq

# Set up the system as a least-squares problem (Xβ = y)
# Given input data about square footage, number of bedrooms, and age of the house
X = np.array([[2100, 3, 20],
              [2500, 4, 15],
              [1800, 2, 30],
              [2200, 3, 25]])

# Given corresponding house prices (in $1000s)
y = np.array([460, 540, 330, 400])

# Solve for β by computing the least-squares solution
beta, residuals, rank, s = lstsq(X, y)

# Display the solution β
print("The coefficients (weights) for square footage, bedrooms, and age are:")
print(beta)

# Use the resulting model to predict the price of a house
# Initiaize input values
sqft = 2400
bedrooms = 3
age = 20

# Create test house
test_house = np.array([sqft, bedrooms, age])

# Calculate predicted price based on the trained model
test_price = np.dot(test_house, beta)
print(f"The predicted price of the house with {sqft} sq ft, {bedrooms} bedrooms, and {age} years old is: ${test_price*1000:.2f}")

