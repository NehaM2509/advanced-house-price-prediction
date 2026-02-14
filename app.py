import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="House Price Prediction", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    return pd.read_csv(url)

data = load_data()

# Keep numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):
    X = numeric_data.drop("median_house_value", axis=1)
    y = numeric_data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, score

model, score = train_model(numeric_data)

# ---------------- HEADER ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏠 House Price Prediction System")
    st.write("Random Forest model trained automatically on housing dataset.")

with col2:
    st.metric("Model R² Score", f"{score:.2f}")

st.markdown("---")

# ---------------- INPUTS ----------------
st.sidebar.header("Enter Property Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("Median Age", 1, 50, 20)
total_rooms = st.sidebar.number_input("Total Rooms", 1, 10000, 2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 5000, 400)
population = st.sidebar.number_input("Population", 1, 5000, 1000)
households = st.sidebar.number_input("Households", 1, 3000, 500)
median_income = st.sidebar.number_input("Median Income", 0.0, 20.0, 5.0)

if st.sidebar.button("Predict Price"):
    input_data = np.array([[longitude, latitude, housing_median_age,
                            total_rooms, total_bedrooms,
                            population, households, median_income]])

    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Price: ${prediction[0]:,.0f}")

st.markdown("---")

# ---------------- HEATMAP ----------------
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("---")
st.markdown("<center>Built using Streamlit & Scikit-learn</center>", unsafe_allow_html=True)
