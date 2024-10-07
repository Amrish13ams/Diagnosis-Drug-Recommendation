import pickle
import argparse

# Load the model
with open('../models/decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Argument parser for input parameters
parser = argparse.ArgumentParser(description='Predict Drug Recommendation.')
parser.add_argument('--age', type=int, required=True, help='Age of the patient')
parser.add_argument('--bp', type=str, required=True, help='Blood pressure level (low, normal, high)')
parser.add_argument('--cholesterol', type=str, required=True, help='Cholesterol level (normal, high)')
parser.add_argument('--na_to_k', type=float, required=True, help='Na to K ratio in the body')

args = parser.parse_args()

# Map categorical inputs to numerical values
bp_mapping = {'low': 0, 'normal': 1, 'high': 2}
cholesterol_mapping = {'normal': 0, 'high': 1}

# Create input for prediction
input_data = [[
    args.age,
    bp_mapping[args.bp.lower()],
    cholesterol_mapping[args.cholesterol.lower()],
    args.na_to_k
]]

# Predict and display the result
prediction = model.predict(input_data)
print(f"Recommended Drug Index: {prediction[0]}")
