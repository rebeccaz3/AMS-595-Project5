import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the loss function
def loss_function(X_flat, A):
    # Reshape X_flat back to the original shape of A
    X = X_flat.reshape(A.shape)
    
    # Loss function
    loss = 0.5 * np.sum((X - A)**2)
    return loss

# Define the gradient (derivative of the loss function)
def gradient(X_flat, A):
    # Reshape X_flat back to the original shape of A
    X = X_flat.reshape(A.shape)
    
    # Gradient (derivative of loss function)
    grad = X - A
    
    # Return the flattened (1D) gradient
    return grad.flatten()

# Implement gradient descent to minimize the loss function
def gradient_descent(A, X_init, threshold=1e-6, max_iter=1000):
    # Initialize list to store the loss values at each iteration
    loss_values = []

    # Callback function to track the loss value at each iteration 
    def callback(X_flat):
        current_loss = loss_function(X_flat, A)
        loss_values.append(current_loss)

    # Use scipy's minimize function to optimize the loss function
    result = minimize(
        fun=loss_function,       # Objective function to minimize
        x0=X_init.flatten(),     # Initial guess for X (1D array)
        jac=gradient,            # Provide jacobian gradient
        args=(A,),               # Arguments that loss_function and gradient can both access
        method='BFGS',           # Optimization method (gradient-based)
        callback=callback,       # Callback function to track loss
        options={'disp': True, 'gtol': threshold, 'maxiter': max_iter}  # Values that help control optimization process
    )
    
    return result, loss_values

# Test Example

# Initialize random matrices X and A
A = np.random.rand(100, 50)  # Random matrix A of shape (100, 50)
X_init = np.random.rand(100, 50)  # Initial guess for X of shape (100, 50)

# Run gradient descent to minimize the loss function
result, loss_values = gradient_descent(A, X_init)

# Print the optimized matrix X (the result of gradient descent)
print("Optimized X matrix:")
print(result.x.reshape(A.shape))  # Reshape the result back to the original shape of A

# Visualize the results (Iterations vs. loss values)
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss Value')
plt.xlabel('Iteration')
plt.ylabel('Loss values')
plt.title('Loss Reduction Over Iterations')
plt.legend()

# Save the plot as a PNG file
plt.savefig('Loss_values.png')

# Display the plot in the command window
plt.show()

