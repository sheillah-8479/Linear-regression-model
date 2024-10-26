# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # For saving the model

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
data = pd.read_csv("accident_data.csv")

# Display the first few rows of the dataset to understand its structure
print(data.head())
# Define dependent and independent variables
# Replace with your actual column names
X = data[['weather_condition', 'road_type', 'traffic_density']]  # Features
y = data['accident_severity']  # Target variable
# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Fill missing values (if any) with mean values for simplicity
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# One-hot encode categorical features, if necessary
X = pd.get_dummies(X, columns=['weather_condition', 'road_type'], drop_first=True)
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes to confirm
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Print the model coefficients and intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Save the trained model to a file
joblib.dump(model, "accident_severity_model.pkl")
print("Model saved as 'accident_severity_model.pkl'")
# Load the saved model
model = joblib.load("accident_severity_model.pkl")

# Create a hypothetical example (replace values with realistic values)
hypothetical_data = pd.DataFrame([[2, 1, 50]], columns=['weather_condition_clear', 'road_type_highway', 'traffic_density'])

# Ensure hypothetical_data has the same features as X after encoding
for col in X.columns:
    if col not in hypothetical_data.columns:
        hypothetical_data[col] = 0  # Add missing columns with default 0 value

# Predict accident severity
predicted_severity = model.predict(hypothetical_data)

print("Predicted Accident Severity:", predicted_severity[0])
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)




