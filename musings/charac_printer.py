import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
# Load the data from the Excel file
file_path = 'FOGRA39.xlsx'  # Replace with the actual file path
data = pd.read_excel(file_path)
# Extract CMYK and LAB values
cmyk = data[['c', 'm', 'y', 'k']].values
lab = data[['l', 'a', 'b']].values
# Determine the split index (80% for training, 20% for testing)
split_index = int(0.8 * len(data))
# Split the data into training and test sets
train_cmyk = cmyk[:split_index]
train_lab = lab[:split_index]
test_cmyk = cmyk[split_index:]
test_lab = lab[split_index:]
# Parameters for Distance-Weighted Interpolation
p = 4.5
epsilon = 0.000001
# Function to compute distance-weighted interpolation
def distance_weighted_interpolation(test_sample, train_samples, train_values):
    distances = np.linalg.norm(train_samples - test_sample, axis=1)
    weights = (1 / (distances + epsilon)) ** p
    weights /= np.sum(weights)
    estimated_value = np.sum(weights[:, None] * train_values, axis=0)
    return estimated_value
# Estimate LAB values for the test set using Distance-Weighted Interpolation
estimated_lab_dwi = []
for test_sample in test_cmyk:
    est_lab = distance_weighted_interpolation(test_sample, train_cmyk, train_lab)
    estimated_lab_dwi.append(est_lab)
# Convert to numpy array
estimated_lab_dwi = np.array(estimated_lab_dwi)
# Calculate Delta E for Distance-Weighted Interpolation
delta_e_dwi = np.linalg.norm(estimated_lab_dwi - test_lab, axis=1)
# Estimate LAB values using Nearest Neighbor Interpolation
estimated_lab_nn = griddata(train_cmyk, train_lab, test_cmyk, method='nearest')
# Calculate Delta E for Nearest Neighbor Interpolation
delta_e_nn = np.linalg.norm(estimated_lab_nn - test_lab, axis=1)
# Estimate LAB values using Linear Interpolation
estimated_lab_linear = griddata(train_cmyk, train_lab, test_cmyk, method='linear')
# Calculate Delta E for Linear Interpolation
delta_e_linear = np.linalg.norm(estimated_lab_linear - test_lab, axis=1)
# Polynomial Fitting (2nd Degree)
poly2 = PolynomialFeatures(degree=2)
train_cmyk_poly2 = poly2.fit_transform(train_cmyk)
test_cmyk_poly2 = poly2.transform(test_cmyk)
# Fit the model
model_poly2 = LinearRegression()
model_poly2.fit(train_cmyk_poly2, train_lab)
# Predict LAB values for the test set using 2nd degree polynomial fitting
estimated_lab_poly2 = model_poly2.predict(test_cmyk_poly2)
# Calculate Delta E for 2nd Degree Polynomial Fitting
delta_e_poly2 = np.linalg.norm(estimated_lab_poly2 - test_lab, axis=1)
# Polynomial Fitting (3rd Degree)
poly3 = PolynomialFeatures(degree=3)
train_cmyk_poly3 = poly3.fit_transform(train_cmyk)
test_cmyk_poly3 = poly3.transform(test_cmyk)
# Fit the model
model_poly3 = LinearRegression()
model_poly3.fit(train_cmyk_poly3, train_lab)
# Predict LAB values for the test set using 3rd degree polynomial fitting
estimated_lab_poly3 = model_poly3.predict(test_cmyk_poly3)
# Calculate Delta E for 3rd Degree Polynomial Fitting
delta_e_poly3 = np.linalg.norm(estimated_lab_poly3 - test_lab, axis=1)
# Optimized Neural Network Model
nn_model = Sequential()
nn_model.add(Dense(128, input_dim=train_cmyk.shape[1], activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.3))
nn_model.add(Dense(256, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.3))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(3, activation='linear'))  # LAB has 3 output nodes
nn_model.compile(optimizer='adam', loss='mean_squared_error')
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the neural network model
nn_model.fit(train_cmyk, train_lab, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
# Predict LAB values for the test set using the optimized Neural Network
estimated_lab_nn_model = nn_model.predict(test_cmyk)
# Calculate Delta E for Neural Network
delta_e_nn_model = np.linalg.norm(estimated_lab_nn_model - test_lab, axis=1)
# Plotting the results
fig, axs = plt.subplots(3, 2, figsize=(14, 15))
# Histogram for Distance-Weighted Interpolation
axs[0, 0].hist(delta_e_dwi, bins=20, color='blue', edgecolor='black')
axs[0, 0].set_title('Distance-Weighted Interpolation')
axs[0, 0].set_xlabel('CIELAB Color Difference (ΔE)')
axs[0, 0].set_ylabel('Frequency')
# Histogram for Nearest Neighbor Interpolation
axs[0, 1].hist(delta_e_nn, bins=20, color='green', edgecolor='black')
axs[0, 1].set_title('Nearest Neighbor Interpolation')
axs[0, 1].set_xlabel('CIELAB Color Difference (ΔE)')
axs[0, 1].set_ylabel('Frequency')
# Histogram for Linear Interpolation
axs[1, 0].hist(delta_e_linear, bins=20, color='orange', edgecolor='black')
axs[1, 0].set_title('Linear Interpolation')
axs[1, 0].set_xlabel('CIELAB Color Difference (ΔE)')
axs[1, 0].set_ylabel('Frequency')
# Histogram for 2nd Degree Polynomial Fitting
axs[1, 1].hist(delta_e_poly2, bins=20, color='purple', edgecolor='black')
axs[1, 1].set_title('2nd Degree Polynomial Fitting')
axs[1, 1].set_xlabel('CIELAB Color Difference (ΔE)')
axs[1, 1].set_ylabel('Frequency')
# Histogram for 3rd Degree Polynomial Fitting
axs[2, 0].hist(delta_e_poly3, bins=20, color='cyan', edgecolor='black')
axs[2, 0].set_title('3rd Degree Polynomial Fitting')
axs[2, 0].set_xlabel('CIELAB Color Difference (ΔE)')
axs[2, 0].set_ylabel('Frequency')
# Histogram for Neural Network
axs[2, 1].hist(delta_e_nn_model, bins=20, color='red', edgecolor='black')
axs[2, 1].set_title('Neural Network')
axs[2, 1].set_xlabel('CIELAB Color Difference (ΔE)')
axs[2, 1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
# Print the mean CIELAB color differences
print(f'Mean CIELAB color difference (Distance-Weighted): {np.mean(delta_e_dwi)}')
print(f'Mean CIELAB color difference (Nearest Neighbor): {np.mean(delta_e_nn)}')
print(f'Mean CIELAB color difference (Linear): {np.mean(delta_e_linear)}')
print(f'Mean CIELAB color difference (2nd Degree Polynomial): {np.mean(delta_e_poly2)}')
print(f'Mean CIELAB color difference (3rd Degree Polynomial): {np.mean(delta_e_poly3)}')
print(f'Mean CIELAB color difference (Neural Network): {np.mean(delta_e_nn_model)}')