import os
import joblib
from preprocess import load_data, split_data
from model import train_model, evaluate_model

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)

    print(f"Test Accuracy: {acc:.2f}")
    print("Classification Report:")
    for label, metrics in report.items():
        print(f"{label}: {metrics}")

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")
    print("Model saved to outputs/model.pkl")

if __name__ == "__main__":
    main()

