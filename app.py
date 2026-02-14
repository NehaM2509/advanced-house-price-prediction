import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
h1 {
    color: #1f4e79;
}
div.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #145a86;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/house_price_model.pkl")
score = joblib.load("models/model_score.pkl")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data/housing.csv")

# ---------------- HEADER SECTION ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏠 Advanced House Price Prediction System")
    st.markdown("""
    Predict house prices using a Random Forest model trained on the Kaggle dataset.
    """)

with col2:
    st.metric("Model R² Score", f"{score:.2f}")

st.markdown("---")

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Sample Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

# ---------------- CORRELATION HEATMAP ----------------
st.subheader("🔍 Correlation Between Selected Features")

selected_cols = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "SalePrice"
]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🏘 Enter House Details")

overall_qual = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Price"):

    input_data = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt, full_bath]])

    prediction = model.predict(input_data)

    # Convert USD to INR
    usd_to_inr = 83
    price_in_inr = prediction[0] * usd_to_inr

    st.markdown("### 💰 Estimated Price")
    st.markdown(f"""
    <div style='background-color:#e6f2ff;padding:20px;border-radius:10px'>
        <h2 style='color:#1f4e79'>₹ {price_in_inr:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📌 Feature Importance")

importances = model.feature_importances_
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.barh(features, importances, color="#1f77b4")
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Built using Streamlit & Scikit-learn</center>",
    unsafe_allow_html=True
)
