#Cellular N model
import numpy as np
from scipy.optimize import minimize

# Function to normalize c, m, y within a cell
def normalize_values(c, m, y, cl, cu, ml, mu, yl, yu):
    c_prime = (c - cl) / (cu - cl)
    m_prime = (m - ml) / (mu - ml)
    y_prime = (y - yl) / (yu - yl)
    return c_prime, m_prime, y_prime

# Function for trilinear interpolation within a given cell
def trilinear_interpolation(c_prime, m_prime, y_prime, P_vertices):
    # P_vertices is a numpy array of shape (8, L) where L is the number of wavelengths
    weights = np.array([(1 - c_prime) * (1 - m_prime) * (1 - y_prime),
                        c_prime * (1 - m_prime) * (1 - y_prime),
                        (1 - c_prime) * m_prime * (1 - y_prime),
                        c_prime * m_prime * (1 - y_prime),
                        (1 - c_prime) * (1 - m_prime) * y_prime,
                        c_prime * (1 - m_prime) * y_prime,
                        (1 - c_prime) * m_prime * y_prime,
                        c_prime * m_prime * y_prime])
    
    R_estimate = np.dot(weights, P_vertices)
    return R_estimate

# Function for spectral regression of Neugebauer primaries
def spectral_regression(R_train, W_train):
    # Compute the least squares solution for P_opt using Equation 5.95
    WtW_inv = np.linalg.inv(W_train.T @ W_train)
    P_opt = WtW_inv @ W_train.T @ R_train
    return P_opt

# Function to compute ΔE metric
def delta_e(predicted, actual):
    return np.linalg.norm(predicted - actual, axis=1).mean()

# Main optimization function
def optimize_neugebauer_model(R_train, CMY_train, P_initial, n_values, alpha_values):
    best_n = None
    best_P = None
    min_delta_e = float('inf')

    for n in n_values:
        # Transform reflectances by Yule-Nielsen factor
        R_train_n = np.power(R_train, 1/n)
        
        # Calculate dot area coverage using Equation 5.92 (not shown here, assuming known values)
        # Placeholder step here: dot_area_function
        W_train = np.random.rand(len(CMY_train), P_initial.shape[0])  # Replace with correct weights

        # Spectral regression to get optimized primaries
        P_opt = spectral_regression(R_train_n, W_train)

        # Calculate ΔE to evaluate model accuracy
        R_pred = W_train @ P_opt
        delta_e_value = delta_e(R_pred, R_train_n)

        # Keep track of best n and P_opt
        if delta_e_value < min_delta_e:
            min_delta_e = delta_e_value
            best_n = n
            best_P = P_opt

    return best_n, np.power(best_P, n)

# Sample inputs
# Define wavelengths and number of samples
num_wavelengths = 31  # Example for visible spectrum from 400-700 nm at 10nm intervals
num_samples = 100  # Number of training samples

# Generate synthetic spectral reflectance data for training (random for demonstration)
R_train = np.random.rand(num_samples, num_wavelengths)  # Reflectance data in range [0, 1]

# Generate synthetic CMY dot area data for training (random for demonstration)
CMY_train = np.random.rand(num_samples, 3)  # Cyan, Magenta, Yellow dot area coverage in range [0, 1]

# Initial Neugebauer primaries (random for demonstration)
P_initial = np.random.rand(8, num_wavelengths)  # 8 vertices for trilinear interpolation

# Range of Yule-Nielsen n values to try
n_values = np.linspace(1, 7, 10)

# Range of alpha values (only applicable if dot-on-dot model is used)
alpha_values = np.linspace(0, 1, 10)

# Optimize the Neugebauer model
best_n, optimized_primaries = optimize_neugebauer_model(R_train, CMY_train, P_initial, n_values, alpha_values)

print("Best Yule-Nielsen factor (n):", best_n)
print("Optimized Neugebauer Primaries (first few values):", optimized_primaries[:3])  # Display first few values
