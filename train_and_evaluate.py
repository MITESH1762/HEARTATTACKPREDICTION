import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import os
import pickle

# Load processed data
data = np.load('data/processed_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'MLP': MLPClassifier(),
    'Deep Learning': None  # To be defined later
}

# Train and evaluate non-deep learning models
results = {}

for name, model in models.items():
    if model is not None:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prob_pred = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, prob_pred)
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC Score': roc_auc
        }

# Define and train deep learning model
def build_deep_learning_model():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

deep_model = build_deep_learning_model()
history = deep_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_dl = (deep_model.predict(X_test) > 0.5).astype(int)
prob_pred_dl = deep_model.predict(X_test)
accuracy_dl = accuracy_score(y_test, y_pred_dl)
precision_dl = precision_score(y_test, y_pred_dl)
recall_dl = recall_score(y_test, y_pred_dl)
f1_dl = f1_score(y_test, y_pred_dl)
roc_auc_dl = roc_auc_score(y_test, prob_pred_dl)

results['Deep Learning'] = {
    'Accuracy': accuracy_dl,
    'Precision': precision_dl,
    'Recall': recall_dl,
    'F1 Score': f1_dl,
    'ROC AUC Score': roc_auc_dl
}

# Ensure the 'static' directory exists
os.makedirs('static', exist_ok=True)

# Plot performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
plt.figure(figsize=(14, 7))
for metric in metrics:
    plt.plot(results.keys(), [results[model][metric] for model in results], label=metric, marker='o')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/model_comparison.png')
plt.show()

# Plot confusion matrix for the best model (e.g., Deep Learning)
best_model = 'Deep Learning'
cm = confusion_matrix(y_test, y_pred_dl)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_model} - Confusion Matrix')
plt.savefig('static/confusion_matrix.png')
plt.show()

# Save the deep learning model
os.makedirs('models', exist_ok=True)  # Ensure the 'models' directory exists
deep_model.save('models/heart_attack_model.h5')

# Optionally save the scaler
with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scale, f)

