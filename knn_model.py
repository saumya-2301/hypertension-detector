import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset from CSV file
data = pd.read_csv('C:/Users/asus/OneDrive - Lovely Professional University/Desktop/hypertension_data.csv')

# Define features and target
X = data[['age', 'gender', 'blood_pressure', 'cholestrol', 'heart_rate', 'exercise']]
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # You can adjust n_neighbors as needed
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# Function to take user input and make predictions
def predict_hypertension():
    age = float(input("Enter age: "))
    gender = int(input("Enter gender (1 for Male, 0 for Female): "))
    blood_pressure = float(input("Enter blood pressure: "))
    cholestrol = float(input("Enter cholesterol level: "))
    heart_rate = float(input("Enter heart rate: "))
    exercise = float(input("Enter exercise (1 for Yes, 0 for No): "))

    # Load the scaler and model
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('knn_model.pkl')

    # Scale the user input
    user_data = np.array([[age, gender, blood_pressure, cholestrol, heart_rate, exercise]])
    user_data_scaled = scaler.transform(user_data)

    # Make a prediction
    prediction = model.predict(user_data_scaled)

    if prediction == 1:
        print("Prediction: The person may suffer from hypertension.")
    else:
        print("Prediction: The person may not suffer from hypertension.")


# Call the function to predict
predict_hypertension()
