import pandas as pd

def preprocess_data(data):
    """
    Function to preprocess the dataset for training or prediction.
    Maps categorical variables to numerical representations.
    """
    # Map categorical variables to numerical values
    data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
    data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})
    return data
