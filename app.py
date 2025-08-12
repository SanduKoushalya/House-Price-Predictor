import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Load model & dataset
# ----------------------------
model = joblib.load("Model/model.pkl")
df = pd.read_csv("data/boston.csv")

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(page_title="Boston Housing Insights", page_icon="🏡", layout="wide")

st.title("🏡 Boston Housing Insights & Prediction")
st.markdown("""
Explore the Boston Housing dataset, make predictions for single or multiple houses,  
and understand **why** the model made those predictions.
""")

# ----------------------------
# Tabs for Navigation
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Prediction", "🔍 Model Explainability"])

# ----------------------------
# TAB 1: Data Exploration
# ----------------------------
with tab1:
    st.subheader("Interactive Data Filtering")

    # Feature filters
    col1, col2, col3 = st.columns(3)
    crim_range = col1.slider("Crime Rate (CRIM)", float(df["CRIM"].min()), float(df["CRIM"].max()), (0.0, 10.0))
    rm_range = col2.slider("Average Rooms (RM)", float(df["RM"].min()), float(df["RM"].max()), (4.0, 8.0))
    lstat_range = col3.slider("Lower Status % (LSTAT)", float(df["LSTAT"].min()), float(df["LSTAT"].max()), (0.0, 20.0))

    filtered_df = df[
        (df["CRIM"].between(crim_range[0], crim_range[1])) &
        (df["RM"].between(rm_range[0], rm_range[1])) &
        (df["LSTAT"].between(lstat_range[0], lstat_range[1]))
    ]

    st.write(f"Filtered Data: {filtered_df.shape[0]} rows")
    st.dataframe(filtered_df)

    # Price distribution
    fig = px.histogram(filtered_df, x="MEDV", nbins=30, title="Distribution of House Prices", color_discrete_sequence=["#2E86C1"])
    st.plotly_chart(fig, use_container_width=True)

    # Rooms vs Price
    fig2 = px.scatter(filtered_df, x="RM", y="MEDV", trendline="ols", title="Rooms vs Price", color="LSTAT")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# TAB 2: Prediction
# ----------------------------
with tab2:
    st.subheader("Predict House Prices")

    pred_type = st.radio("Choose Prediction Type", ["Single Entry", "Bulk Upload"])

    if pred_type == "Single Entry":
        CRIM = st.number_input("Per capita crime rate (CRIM)", value=0.1)
        RM = st.number_input("Average number of rooms (RM)", value=6.0)
        AGE = st.number_input("Proportion of owner-occupied units built before 1940 (AGE)", value=65.0)
        TAX = st.number_input("Property tax rate (TAX)", value=300)
        LSTAT = st.number_input("Lower status population % (LSTAT)", value=12.0)

        features = np.array([[CRIM, RM, AGE, TAX, LSTAT]])
        prediction = model.predict(features)
        st.success(f"Predicted Price: ${prediction[0]*1000:.2f}")

    else:
        uploaded_file = st.file_uploader("Upload CSV file with columns: CRIM, RM, AGE, TAX, LSTAT", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            predictions = model.predict(input_df)
            input_df["Predicted Price ($)"] = predictions * 1000
            st.dataframe(input_df)
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# ----------------------------
# TAB 3: Explainability
# ----------------------------
with tab3:
    st.subheader("Model Explainability with SHAP")
    st.markdown("Understand the contribution of each feature to the prediction.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[["CRIM", "RM", "AGE", "TAX", "LSTAT"]])

    st.write("### Feature Importance (Global)")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, df[["CRIM", "RM", "AGE", "TAX", "LSTAT"]], plot_type="bar", show=False)
    st.pyplot(fig1)
    plt.clf()  # Clear figure after showing

    st.write("### Detailed Impact (Summary Plot)")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, df[["CRIM", "RM", "AGE", "TAX", "LSTAT"]], show=False)
    st.pyplot(fig2)
    plt.clf()  # Clear figure after showing
