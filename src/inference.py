import joblib
import numpy as np

def load_model(model_path="outputs/model.pkl"):
    """
    Load the trained model from a file
    """
    return joblib.load(model_path)

def predict(model, sample):
    """
    Make a prediction on a single sample
    sample: list or np.array of shape (4,) representing flower features
    """
    sample = np.array(sample).reshape(1, -1)  # Reshape the sample to match model input
    prediction = model.predict(sample)[0]
    return prediction

if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Example input: [sepal_length, sepal_width, petal_length, petal_width]
    sample = [5.1, 3.5, 1.4, 0.2]  # Example flower measurements

    # Predict the flower species
    result = predict(model, sample)
    class_names = ["Setosa", "Versicolor", "Virginica"]
    print(f" Predicted Species: {class_names[result]}")
    print(f" Predicted Class Index: {result}")