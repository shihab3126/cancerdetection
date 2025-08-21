from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from collections import Counter

app = Flask(__name__)

# Decision Tree Class Definition (same as used for training)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If we reached maximum depth or pure leaf node
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return unique_classes[0]

        # Finding the best feature and threshold for splitting
        best_gini = float('inf')
        best_split = None
        for feature_index in range(num_features):
            # Make sure there's data to iterate over thresholds
            if not X.iloc[:, feature_index].empty:
                thresholds = np.unique(X.iloc[:, feature_index])
                for threshold in thresholds:
                    left_mask = X.iloc[:, feature_index] <= threshold
                    right_mask = ~left_mask
                    # Ensure masks are not empty before indexing
                    if not y[left_mask].empty and not y[right_mask].empty:
                        left_y = y[left_mask]
                        right_y = y[right_mask]

                        # Gini impurity calculation
                        left_gini = 1 - sum([(np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(left_y)])
                        right_gini = 1 - sum([(np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(right_y)])
                        gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)

                        if gini < best_gini:
                            best_gini = gini
                            best_split = (feature_index, threshold, left_mask, right_mask)
        # Handle the case where no split improves Gini
        if best_split is None:
            # Return the most frequent class in the current node
            return Counter(y).most_common(1)[0][0]

        # Recursively build the tree
        feature_index, threshold, left_mask, right_mask = best_split
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X.values]

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature_index']] <= tree['threshold']:
                return self._predict_one(x, tree['left'])
            else:
                return self._predict_one(x, tree['right'])
        else:
            return tree

# Load the trained model from the uploaded pickle file
model_filename = 'cancer_model.pkl'  # Ensure this is the correct path where your model is saved
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# Define the columns expected in the input form
input_columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values from the form
    try:
        features = [float(request.form[column]) for column in input_columns]
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid values.")

    # Convert features to a DataFrame
    input_data = pd.DataFrame([features], columns=input_columns)

    # Predict the class using the model
    prediction = model.predict(input_data)[0]

    # Display the result
    if prediction == 1:
        prediction_text = "The person is likely to have cancer."
    else:
        prediction_text = "The person is likely to be cancer-free."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
