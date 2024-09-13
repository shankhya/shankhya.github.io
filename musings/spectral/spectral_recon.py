
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast


# Load the dataset
file_path = 'spectraldb.csv'  # Replace with the actual file path
spectral_data = pd.read_csv(file_path)

# Step 1: Remove rows with missing SCIMeasures or LAB values
spectral_data_cleaned = spectral_data.dropna(subset=['SCIMeasures', 'L', 'a', 'b']).copy()

# Step 2: Convert 'SCIMeasures' (which is a string representation of a dictionary) to a sorted list of spectral values
def extract_spectral_data(spectral_column):
    # Convert string representation of dictionary to an actual dictionary and sort by wavelength
    return spectral_column.apply(lambda x: {int(k): v for k, v in sorted(ast.literal_eval(x).items())})

# Apply the function to extract SCIMeasures
spectral_data_cleaned['SCIMeasures'] = extract_spectral_data(spectral_data_cleaned['SCIMeasures'])

# Step 3: Filter the data to only include samples where the number of spectral measurements is 39
spectral_data_filtered = spectral_data_cleaned[spectral_data_cleaned['SCIMeasures'].apply(len) == 39].copy()

# Extract relevant information: Sample ID and Name
sample_id = spectral_data_filtered['ID']
sample_name = spectral_data_filtered['Name']

# Extract the SCIMeasures and create a DataFrame with 39 wavelengths
wavelengths_39 = sorted(list(spectral_data_filtered['SCIMeasures'].iloc[0].keys()))
spectral_df_39 = pd.DataFrame(columns=wavelengths_39)

# Iterate through each row and add the spectral values for the 39 wavelengths
for i, row in spectral_data_filtered.iterrows():
    spectral_values = row['SCIMeasures']
    row_data = {wavelength: spectral_values[wavelength] for wavelength in wavelengths_39}
    spectral_df_39.loc[i] = row_data.values()

# Add the Sample ID, Name, and LAB values
spectral_df_39.insert(0, 'Sample ID', sample_id)
spectral_df_39.insert(1, 'Sample Name', sample_name)
spectral_df_39[['L', 'a', 'b']] = spectral_data_filtered[['L', 'a', 'b']]

# Save the filtered DataFrame to a CSV file
output_file_filtered = 'refined_spectral_data.csv'
spectral_df_39.to_csv(output_file_filtered, index=False)

print(f"Filtered data saved to {output_file_filtered}")


# Load the filtered dataset with 39 spectral measurements per sample
file_path = 'refined_spectral_data.csv'  # Path to your new filtered CSV file
spectral_data = pd.read_csv(file_path)

# Extract the spectral data (wavelengths) and LAB values
# Assuming the spectral data is in the first 39 columns, and LAB values are in the last 3 columns
X = spectral_data[['L', 'a', 'b']].values  # LAB values as input
y = spectral_data.iloc[:, 2:41].values  # Spectral values as target (assuming they are in columns 3 to 41)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize LAB values (input)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize spectral data (output)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Define the model with an explicit Input layer
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer 1
    Dense(128, activation='relu'),  # Hidden layer 2
    Dense(256, activation='relu'),  # Hidden layer 3
    Dense(y_train_scaled.shape[1])  # Output layer for 39 spectral values
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and save training history
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=16, validation_split=0.1)

# Evaluate the model on test data
loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test MSE: {loss}")

# Predict the spectral data using the trained model
y_pred_scaled = model.predict(X_test_scaled)

# Inverse transform the predicted and test data to get the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Calculate metrics for both training and testing data
y_train_pred_scaled = model.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_train_original = scaler_y.inverse_transform(y_train_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train_original, y_train_pred)
test_mse = mean_squared_error(y_test_original, y_pred)
train_mape = mean_absolute_percentage_error(y_train_original, y_train_pred)
test_mape = mean_absolute_percentage_error(y_test_original, y_pred)
train_r2 = r2_score(y_train_original, y_train_pred)
test_r2 = r2_score(y_test_original, y_pred)

# Display the results
print(f"Training MSE: {train_mse}, Testing MSE: {test_mse}")
print(f"Training MAPE: {train_mape}, Testing MAPE: {test_mape}")
print(f"Training R2: {train_r2}, Testing R2: {test_r2}")

# Plot training & validation loss values (only the loss)
plt.figure(figsize=(8, 6))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend(loc='upper right')

plt.show()

# Select 20 random colors from the test set
num_samples = 20
random_indices = np.random.choice(len(y_test_original), num_samples, replace=False)  # Randomly select 20 indices
y_pred_sample = y_pred[random_indices]  # Predicted spectral data for the selected 20 samples
y_test_original_sample = y_test_original[random_indices]  # Actual spectral data for the selected 20 samples

# Get the Sample IDs and Sample Names for the selected 20 samples
sample_ids = spectral_data['Sample ID'].values[random_indices]
sample_names = spectral_data['Sample Name'].values[random_indices]

# Define the correct wavelengths for the 39 spectral measurements
wavelengths = np.linspace(360, 780, 39)

# Create the plot with 4 columns and 5 rows (20 plots in total)
fig, axes = plt.subplots(4, 5, figsize=(18, 20))

# Reduce font size for the ticks and labels
plt.rc('font', size=10)

for i in range(num_samples):
    row = i // 5
    col = i % 5
    axes[row, col].plot(wavelengths, y_test_original_sample[i], label='Actual', color='blue')
    axes[row, col].plot(wavelengths, y_pred_sample[i], label='Predicted', linestyle='dashed', color='red')
    axes[row, col].set_xlabel('Wavelength (nm)', fontsize=8)
    axes[row, col].set_ylabel('Reflectance', fontsize=8)
    axes[row, col].tick_params(axis='both', which='major', labelsize=6)  # Reduce tick font size
    axes[row, col].text(0.5, -0.15, f"ID: {sample_ids[i]}, Name: {sample_names[i]}",
                        fontsize=8, ha='center', transform=axes[row, col].transAxes)

# Create a single legend for the entire plot in the top-right corner
lines, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one plot
fig.legend(lines, labels, loc='upper right', fontsize=8)


# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()


# Save the model in Keras format
model.save('spectral_reconstruction_model.keras')

# Load the model
loaded_model = tf.keras.models.load_model('spectral_reconstruction_model.keras')

# Compile the model after loading, if needed
loaded_model.compile(optimizer='adam', loss='mean_squared_error')

# View the summary of the model
model.summary()