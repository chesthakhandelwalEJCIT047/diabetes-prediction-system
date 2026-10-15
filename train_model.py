import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Separate features and target
X = data.drop(columns="Outcome")
y = data["Outcome"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=2
)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
model = svm.SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

# Evaluate the model
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Save the feature names and order
with open("feature_names.pkl", "wb") as file:
    pickle.dump(list(X.columns), file)

print("Model, scaler, and feature names saved successfully!")