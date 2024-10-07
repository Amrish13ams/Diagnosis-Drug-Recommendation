# Diagnosis and Drug Recommendation Using Decision Tree Classifier

This project aims to predict the appropriate drug recommendation for a patient based on their symptoms, such as blood pressure (BP) levels, cholesterol levels, and other health indicators, using a decision tree classifier.

## Dataset
The dataset contains features like:
- Age
- BP level (High, Normal, Low)
- Cholesterol level (High, Normal)
- Na to K ratio in the body
- Patient gender, etc.

The model predicts the drug index that is most likely to benefit the patient.

## Algorithms Used
- Decision Tree Classifier

## How to Run the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Diagnosis-Drug-Recommendation.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the model training script:
    ```bash
    python scripts/train_model.py
    ```
4. Make predictions using:
    ```bash
    python scripts/predict.py --age 45 --bp high --cholesterol normal --na_to_k 12.5
    ```

## Dependencies
- scikit-learn
- pandas
- numpy

## License
MIT License
