import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

file_path = r"adas_svm_sorted.csv"  # Full path to the file
data = pd.read_csv(file_path)

# Prepare features and labels
X = data[['Avg_X', 'Avg_Y','Avg_Distance']]  # Changed column names
y = data['Label']                             # Changed column name

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for NaN or infinite values
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    raise ValueError("Input data contains NaN or infinite values.")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib

# Save the model and scaler
joblib.dump(svm, "./svm_model.pkl")  # Save SVM model
joblib.dump(scaler, "./scaler.pkl")  # Save the scaler
print("Model and scaler saved successfully!")