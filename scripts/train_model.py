import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load and preprocess data
data = pd.read_csv('../data/drug_data.csv')
data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

X = data[['Age', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('../models/decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")
