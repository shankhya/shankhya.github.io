import numpy as np
import pandas as pd

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Colour science library
import colour
from colour import (
    SDS_ILLUMINANTS,
    SpectralShape,
    MSDS_CMFS,
    XYZ_to_CIECAM02,
    SpectralDistribution,
)

# Visualization libraries
import matplotlib.pyplot as plt

# Step 1: Load Spectral Reflectance Data and XYZ Values

# Load the spectral reflectance data and XYZ values
spectral_df = pd.read_csv('X.csv', header=0)  # Spectral data (wavelengths as index)
xyz_df = pd.read_csv('XYZ.csv', header=0)  # XYZ data

# Transpose the spectral data so that each row represents a color patch
# After transposing, each row in spectral_df corresponds to a color patch
spectral_df = spectral_df.transpose()

# Verify the shapes after transposition
print(f"Spectral data shape after transposition: {spectral_df.shape}")  # Expected: (1600, 401)
print(f"XYZ data shape: {xyz_df.shape}")  # Expected: (1600, 3)

# Ensure both DataFrames have the same number of color patches (1600 rows)
assert spectral_df.shape[0] == xyz_df.shape[0], "Mismatch in number of color patches."

# Step 2: Simulate Various Viewing Conditions

# Define viewing conditions
illuminant_names = ['D65', 'A', 'FL1']  # Standard illuminants
luminance_levels = [31.83, 318.3, 3183]  # cd/m^2 (low to high)
backgrounds = ['Gray', 'White', 'Black']

# Prepare lists to store data
data_entries = []

# Standard observer color matching functions
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy()
cmfs = cmfs.align(SpectralShape(380, 780, 1))

# Iterate over viewing conditions
for illuminant_name in illuminant_names:
    # Get illuminant SPD
    illuminant = SDS_ILLUMINANTS[illuminant_name].copy()
    illuminant = illuminant.align(SpectralShape(380, 780, 1))

    # Compute XYZ of the illuminant
    XYZ_w = colour.sd_to_XYZ(illuminant, cmfs)
    Y_w = XYZ_w[1]

    for luminance in luminance_levels:
        for background in backgrounds:
            # Set background Y_b based on background reflectance
            if background == 'Gray':
                Y_b = Y_w * 0.20  # 20% reflectance
            elif background == 'White':
                Y_b = Y_w * 0.80  # 80% reflectance
            elif background == 'Black':
                Y_b = Y_w * 0.05  # 5% reflectance
            else:
                Y_b = Y_w * 0.20  # Default to Gray if unknown

            L_A = luminance / 5  # Simplified adaptation luminance

            # Define surround condition (Average, Dim, or Dark)
            if luminance >= 318.3:
                surround = colour.VIEWING_CONDITIONS_CIECAM02['Average']
            elif luminance >= 31.83:
                surround = colour.VIEWING_CONDITIONS_CIECAM02['Dim']
            else:
                surround = colour.VIEWING_CONDITIONS_CIECAM02['Dark']

            # Iterate over each color patch
            for i in range(len(spectral_df)):
                # Get spectral reflectance for the color patch
                reflectance_values = spectral_df.iloc[i].values
                reflectance_sd = SpectralDistribution(
                    data=reflectance_values,
                    domain=spectral_df.columns.astype(float),
                    name=f'Patch {i}'
                )

                # Compute XYZ of the color patch under the illuminant
                XYZ = colour.sd_to_XYZ(
                    sd=reflectance_sd,
                    illuminant=illuminant,
                    cmfs=cmfs
                )

                # Normalize XYZ by luminance
                XYZ = XYZ / Y_w * luminance

                # Compute perceptual attributes using CIECAM02
                specification = XYZ_to_CIECAM02(
                    XYZ=XYZ,
                    XYZ_w=XYZ_w,
                    L_A=L_A,
                    Y_b=Y_b,
                    surround=surround
                )

                # Collect data
                entry = {
                    'Patch': spectral_df.index[i],
                    'Illuminant': illuminant_name,
                    'Luminance': luminance,
                    'Background': background,
                    'X': XYZ[0],
                    'Y': XYZ[1],
                    'Z': XYZ[2],
                    'J': specification.J,
                    'C': specification.C,
                    'h': specification.h,
                    's': specification.s,
                    'Q': specification.Q,
                    'M': specification.M,
                    'H': specification.H
                }
                data_entries.append(entry)

        print(f"Data collection completed for illuminant {illuminant_name} at luminance {luminance}.")

print("All data collection completed.")

# Step 3: Prepare the Dataset for Machine Learning

# Create DataFrame
data = pd.DataFrame(data_entries)

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Illuminant', 'Background'])

# Define features and target variables
features = data_encoded.drop(columns=['Patch', 'J', 'C', 'h', 's', 'Q', 'M', 'H'])
targets = data_encoded[['J', 'C', 'h', 's', 'Q', 'M', 'H']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train a Neural Network to Predict Perceptual Attributes

# Initialize neural network regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# Train the model
print("Training the neural network...")
mlp.fit(X_train_scaled, y_train)
print("Model training completed.")

# Step 5: Evaluate and Analyze the Model's Performance

# Predict on test data
y_pred = mlp.predict(X_test_scaled)

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R^2 Score (overall): {r2:.4f}")
print(f"Model Mean Squared Error (overall): {mse:.4f}")

# For individual attributes
attributes = ['J', 'C', 'h', 's', 'Q', 'M', 'H']
for idx, attr in enumerate(attributes):
    r2_attr = r2_score(y_test[attr], y_pred[:, idx])
    mse_attr = mean_squared_error(y_test[attr], y_pred[:, idx])
    print(f"Attribute: {attr} - R^2 Score: {r2_attr:.4f}, MSE: {mse_attr:.4f}")

# Visualize Actual vs Predicted for one attribute, e.g., Lightness (J)
plt.figure(figsize=(8,6))
plt.scatter(y_test['J'], y_pred[:, attributes.index('J')], alpha=0.5)
plt.xlabel('Actual J')
plt.ylabel('Predicted J')
plt.title('Actual vs Predicted Lightness (J)')
plt.plot([y_test['J'].min(), y_test['J'].max()], [y_test['J'].min(), y_test['J'].max()], 'r--')
plt.show()

# Step 6: Save the Trained Model (Optional)

import joblib

# Save the model
joblib.dump(mlp, 'color_appearance_model.pkl')

# Load the model (when needed)
# mlp_loaded = joblib.load('color_appearance_model.pkl')

print("Model evaluation and saving completed.")