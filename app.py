import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    return pd.read_csv(url)

data = load_data()

# Keep only numeric columns
numeric_data = data.select_dtypes(include=["float64", "int64"])

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    score = r2_score(y_test, model.predict(X_test))
    return model, score

model, score = train_model(numeric_data)

# ---------------- HEADER ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏠 House Price Prediction System")
    st.write("Random Forest model trained automatically on California Housing dataset.")

with col2:
    st.metric("Model R² Score", f"{score:.2f}")

st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🏘 Enter Property Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("Median Age", 1, 50, 20)
total_rooms = st.sidebar.number_input("Total Rooms", 1, 10000, 2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 5000, 400)
population = st.sidebar.number_input("Population", 1, 5000, 1000)
households = st.sidebar.number_input("Households", 1, 3000, 500)
median_income = st.sidebar.number_input("Median Income", 0.0, 20.0, 5.0)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Price"):

    input_data = np.array([[longitude, latitude, housing_median_age,
                            total_rooms, total_bedrooms,
                            population, households, median_income]])

    prediction = model.predict(input_data)

    # Convert USD to INR
    usd_to_inr = 83
    price_in_inr = prediction[0] * usd_to_inr

    st.markdown("### 💰 Estimated Price")
    st.markdown(
        f"""
        <div style='background-color:#e6f2ff;padding:20px;border-radius:10px'>
            <h2 style='color:#1f4e79'>₹ {price_in_inr:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------- HEATMAP ----------------
st.subheader("📊 Correlation Heatmap (Top Influential Features)")

corr = numeric_data.corr()
top_features = corr["median_house_value"].abs().sort_values(ascending=False).head(8).index
filtered_corr = corr.loc[top_features, top_features]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

st.markdown("---")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📌 Feature Importance")

feature_names = numeric_data.drop("median_house_value", axis=1).columns
importances = model.feature_importances_

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(feature_names, importances, color="#1f77b4")
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)

st.markdown("---")
st.markdown(
    "<center>Built using Streamlit & Scikit-learn</center>",
    unsafe_allow_html=True
)
