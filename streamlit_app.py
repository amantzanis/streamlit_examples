import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.joblib')

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Streamlit app
st.title("Iris Classification Model")

# Sidebar for user inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(iris_df['sepal length (cm)'].min()), float(iris_df['sepal length (cm)'].max()), float(iris_df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(iris_df['sepal width (cm)'].min()), float(iris_df['sepal width (cm)'].max()), float(iris_df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(iris_df['petal length (cm)'].min()), float(iris_df['petal length (cm)'].max()), float(iris_df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(iris_df['petal width (cm)'].min()), float(iris_df['petal width (cm)'].max()), float(iris_df['petal width (cm)'].mean()))

# Make predictions
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

st.sidebar.header("Prediction")
st.sidebar.write(f"Predicted species: {iris.target_names[prediction][0]}")
st.sidebar.write("Prediction probabilities:")
for i, species in enumerate(iris.target_names):
    st.sidebar.write(f"{species}: {prediction_proba[0][i]:.2f}")

# Display dataset
st.header("Dataset")
st.write(iris_df.head())

# Display dataset statistics
st.header("Dataset Statistics")
st.write(iris_df.describe())

# Visualizations
st.header("Visualizations")

# Pairplot
st.subheader("Pairplot")
sns.pairplot(iris_df, hue='species')
st.pyplot(plt)