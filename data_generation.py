import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Seed for reproducibility
np.random.seed(42)

# Number of samples per class
n_samples_per_class = 500

# Generate synthetic data for class 0
age_0 = np.random.randint(30, 80, n_samples_per_class)
gender_0 = np.random.randint(0, 2, n_samples_per_class)  # 0: Female, 1: Male
family_history_0 = np.random.randint(0, 2, n_samples_per_class)
smoking_0 = np.random.randint(0, 2, n_samples_per_class)
alcohol_0 = np.random.randint(0, 2, n_samples_per_class)
physical_activity_0 = np.random.randint(0, 2, n_samples_per_class)
diet_0 = np.random.randint(0, 2, n_samples_per_class)  # 0: Unhealthy, 1: Healthy
bmi_0 = np.random.normal(25, 5, n_samples_per_class)
blood_pressure_0 = np.random.normal(120, 15, n_samples_per_class)
cholesterol_0 = np.random.normal(200, 30, n_samples_per_class)
diabetes_0 = np.random.randint(0, 2, n_samples_per_class)
stress_0 = np.random.randint(0, 2, n_samples_per_class)
heart_attack_0 = np.zeros(n_samples_per_class)  # Class 0

# Generate synthetic data for class 1
age_1 = np.random.randint(30, 80, n_samples_per_class)
gender_1 = np.random.randint(0, 2, n_samples_per_class)  # 0: Female, 1: Male
family_history_1 = np.random.randint(0, 2, n_samples_per_class)
smoking_1 = np.random.randint(0, 2, n_samples_per_class)
alcohol_1 = np.random.randint(0, 2, n_samples_per_class)
physical_activity_1 = np.random.randint(0, 2, n_samples_per_class)
diet_1 = np.random.randint(0, 2, n_samples_per_class)  # 0: Unhealthy, 1: Healthy
bmi_1 = np.random.normal(25, 5, n_samples_per_class)
blood_pressure_1 = np.random.normal(120, 15, n_samples_per_class)
cholesterol_1 = np.random.normal(200, 30, n_samples_per_class)
diabetes_1 = np.random.randint(0, 2, n_samples_per_class)
stress_1 = np.random.randint(0, 2, n_samples_per_class)
heart_attack_1 = np.ones(n_samples_per_class)  # Class 1

# Combine data from both classes
age = np.concatenate([age_0, age_1])
gender = np.concatenate([gender_0, gender_1])
family_history = np.concatenate([family_history_0, family_history_1])
smoking = np.concatenate([smoking_0, smoking_1])
alcohol = np.concatenate([alcohol_0, alcohol_1])
physical_activity = np.concatenate([physical_activity_0, physical_activity_1])
diet = np.concatenate([diet_0, diet_1])
bmi = np.concatenate([bmi_0, bmi_1])
blood_pressure = np.concatenate([blood_pressure_0, blood_pressure_1])
cholesterol = np.concatenate([cholesterol_0, cholesterol_1])
diabetes = np.concatenate([diabetes_0, diabetes_1])
stress = np.concatenate([stress_0, stress_1])
heart_attack = np.concatenate([heart_attack_0, heart_attack_1])

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'FamilyHistory': family_history,
    'Smoking': smoking,
    'Alcohol': alcohol,
    'PhysicalActivity': physical_activity,
    'Diet': diet,
    'BMI': bmi,
    'BloodPressure': blood_pressure,
    'Cholesterol': cholesterol,
    'Diabetes': diabetes,
    'Stress': stress,
    'HeartAttack': heart_attack
})

# Check overall class distribution
print("Overall class distribution:")
print(data['HeartAttack'].value_counts())

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save the data to a CSV file
data.to_csv('data/heart_attack_data.csv', index=False)

# Split data into features and target
X = data.drop('HeartAttack', axis=1)
y = data['HeartAttack']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in training set
print("Class distribution in training data:")
print(y_train.value_counts())

# Ensure both classes are present in training data
if len(y_train.value_counts()) < 2:
    raise ValueError("Training data does not contain samples from both classes. Please check the data generation process.")

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the processed data and scaler
np.savez('data/processed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import numpy as np
import os

# Define and train your model
def build_deep_learning_model():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming X_train and y_train are defined
deep_model = build_deep_learning_model()
deep_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save the model
deep_model.save('models/heart_attack_model.h5')
