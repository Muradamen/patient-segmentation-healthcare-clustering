import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Patient Segmentation Dashboard", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data_from_file(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_data_from_upload(uploaded_file):
    return pd.read_csv(uploaded_file)


st.title("🏥 Patient Segmentation & Prediction Dashboard")

df = None

# -------------------------------
# CHECK LOCAL FILE FIRST
# -------------------------------
if os.path.exists("clustered_patients.csv"):
    df = load_data_from_file("clustered_patients.csv")
    st.success("Loaded dataset from repository ✅")

    use_upload = st.checkbox("📂 Upload a different dataset")

    if use_upload:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = load_data_from_upload(uploaded_file)

else:
    st.warning("No dataset found in repo. Please upload one.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = load_data_from_upload(uploaded_file)
    else:
        st.stop()

# -------------------------------
# DATA OVERVIEW
# -------------------------------
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

with col2:
    st.write("Data Types")
    st.write(df.dtypes)

# -------------------------------
# CLUSTER LABELS
# -------------------------------
cluster_names = {
    1: "High-Cost Chronic Patients",
    2: "Frequent Healthcare Users",
    3: "Low-Risk Young Patients",
    4: "Inactive Patients"
}

df["Segment"] = df["Cluster"].map(cluster_names)

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📊 Cluster Distribution")

fig1 = px.pie(
    df,
    names="Segment",
    title="Patient Segments",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💰 Billing Behavior")

fig2 = px.box(
    df,
    x="Segment",
    y="Avg_Billing_Amount",
    color="Segment",
    title="Billing Distribution by Segment"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# FEATURE ANALYSIS
# -------------------------------
st.subheader("🔍 Feature Analysis")

feature = st.selectbox(
    "Select Feature",
    ["Age", "BMI", "Annual_Visits", "Days_Since_Last_Visit"]
)

fig3 = px.histogram(
    df,
    x=feature,
    color="Segment",
    barmode="overlay",
    marginal="box",
    title=f"{feature} Distribution"
)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# PREDICTION SYSTEM
# -------------------------------
st.sidebar.header("🧠 Predict Patient Segment")

numeric_cols = [
    'Age', 'BMI', 'Num_Chronic_Conditions',
    'Annual_Visits', 'Avg_Billing_Amount',
    'Days_Since_Last_Visit'
]

categorical_cols = [
    'Gender', 'Insurance_Type', 'Primary_Condition'
]

def predict_cluster(new_data, df_ref):
    scaler = MinMaxScaler()
    scaler.fit(df_ref[numeric_cols])

    df_num_scaled = scaler.transform(df_ref[numeric_cols])
    new_num_scaled = scaler.transform(pd.DataFrame([new_data])[numeric_cols])

    cat_dist = np.column_stack([
        (df_ref[col].values != new_data[col]).astype(float)
        for col in categorical_cols
    ])

    num_dist = np.abs(df_num_scaled - new_num_scaled)
    gower_dist = np.concatenate([num_dist, cat_dist], axis=1).mean(axis=1)

    cluster_labels = df_ref['Cluster']

    cluster_dist = {
        cluster: gower_dist[cluster_labels == cluster].mean()
        for cluster in np.unique(cluster_labels)
    }

    return min(cluster_dist, key=cluster_dist.get)

with st.sidebar.form("prediction_form"):
    age = st.number_input("Age", 0, 100, 40)
    gender = st.selectbox("Gender", df["Gender"].unique())
    insurance = st.selectbox("Insurance Type", df["Insurance_Type"].unique())
    condition = st.selectbox("Primary Condition", df["Primary_Condition"].unique())
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    chronic = st.slider("Chronic Conditions", 0, 10, 1)
    visits = st.number_input("Annual Visits", 0, 50, 5)
    billing = st.number_input("Avg Billing", 0, 10000, 3000)
    days = st.number_input("Days Since Last Visit", 0, 365, 100)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Age': age,
        'BMI': bmi,
        'Num_Chronic_Conditions': chronic,
        'Annual_Visits': visits,
        'Avg_Billing_Amount': billing,
        'Days_Since_Last_Visit': days,
        'Gender': gender,
        'Insurance_Type': insurance,
        'Primary_Condition': condition
    }

    cluster = predict_cluster(input_data, df)
    segment = cluster_names.get(cluster, f"Cluster {cluster}")

    st.sidebar.success(f"Predicted Segment: {segment}")

# -------------------------------
# DOWNLOAD
# -------------------------------
st.subheader("⬇️ Download Results")

st.download_button(
    "Download Dataset",
    data=df.to_csv(index=False),
    file_name="clustered_patients.csv",
    mime="text/csv"
)