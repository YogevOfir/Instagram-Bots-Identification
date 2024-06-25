import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load the datasets and preprocess them
train_data = pd.read_csv('train.csv')
X_train = train_data.drop(columns=['fake'])
y_train = train_data['fake']

test_data = pd.read_csv('test.csv')
X_test = test_data.drop(columns=['fake'])
y_test = test_data['fake']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and their corresponding names
models = [
    ("KNN", KNeighborsClassifier(n_neighbors=7, p=1)),
    ("SVM", SVC(C=0.1, kernel='linear')),
    ("AdaBoost", AdaBoostClassifier(SVC(C=1, kernel='linear'), n_estimators=100, algorithm='SAMME', random_state=42)),
    ("Logistic Regression", LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000))
]

# Train each model and calculate true error
results = []
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    true_error = mean_squared_error(y_test, y_test_pred)
    results.append((name, true_error))

# Plot the results
model_names, errors = zip(*results)
plt.figure(figsize=(10, 6))
plt.bar(model_names, errors, color='skyblue')
plt.xlabel('Model')
plt.ylabel('True Error')
plt.title('Comparison of Machine Learning Models')
plt.ylim(0, max(errors) + 0.1)  # Adjust y-axis limit for better visualization
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
