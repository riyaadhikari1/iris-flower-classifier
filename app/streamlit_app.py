import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return joblib.load("outputs/model.pkl")

# Load the trained model
model = load_model()

# Class names corresponding to predicted labels
class_names = ["Setosa", "Versicolor", "Virginica"]

# Function to load images for each class
def load_flower_image(class_name):
    image_path = f"images/{class_name.lower()}.jpg"
    return Image.open(image_path)

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower's measurements below to predict the species.")

# Sliders for user inputs (flower measurements)
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# CSV file upload for batch prediction
st.subheader("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded data:")
    st.dataframe(df)

    if st.button("Predict Batch"):
        predictions = model.predict(df)
        df['Predicted Species'] = [class_names[pred] for pred in predictions]
        st.write("Prediction Results:")
        st.dataframe(df)

# Prediction button
if st.button("Predict"):
    sample = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = model.predict(pd.DataFrame([sample], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']))[0]
    prediction_proba = model.predict_proba(pd.DataFrame([sample], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']))[0]
    
    st.success(f"🌸 Predicted Species: **{class_names[prediction]}**")
    
    # Show confidence score
    st.write(f"Confidence Score: **{prediction_proba[prediction] * 100:.2f}%**")

    # Display probability distribution for all classes
    st.write("Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction_proba[i] * 100:.2f}%")
    
    # Display image of predicted class
    image = load_flower_image(class_names[prediction])
    st.image(image, caption=class_names[prediction], use_column_width=True)
