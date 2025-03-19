



import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
X_train = np.sort(np.random.rand(10, 1) * 10, axis=0)  # 10 random points between 0 and 10
y_train = np.sin(X_train) + np.random.normal(0, 0.1, X_train.shape)  # Sinusoidal function with noise

# Plot the training data
plt.scatter(X_train, y_train, color='red', label='Training Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Data: Noisy Sinusoidal')
plt.show()


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the kernel (RBF kernel with length scale of 1)
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

# Create the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to the noisy training data
gp.fit(X_train, y_train)

# Make predictions at new test points
X_test = np.linspace(0, 10, 20).reshape(-1, 1)  # Test data (points to predict)
y_pred, sigma = gp.predict(X_test, return_std=True)  # Predict the mean and standard deviation

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='red', label='Training Data')
# plt.scatter(X_test, y_pred, color='blue', marker='*', label='GP Prediction (mean)')
plt.plot(X_test, y_pred, color='blue', marker='*', label='GP Prediction (mean)')
# plt.fill_between(X_test.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue', label='95% Confidence Interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression (GP) for Noisy Sinusoidal Data')
plt.legend()
plt.show()

