import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("random_forest_ids.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
threshold = pickle.load(open("threshold.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

st.set_page_config(
    page_title="AI-Based Intrusion Detection System",
    page_icon="🔐",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>🔐 AI-Based Intrusion Detection System</h1>
    <p style='text-align:center; font-size:18px;'>
    Random Forest based Network Traffic Classification
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📌 Project Information")
st.sidebar.markdown(
    """
    **Domain:** Cybersecurity  
    **Technique:** Supervised Learning  
    **Model:** Random Forest  
    **Dataset:** NSL-KDD  
    **Classes:** Benign / Malicious  
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Enter Network Parameters")

def user_input_features():
    duration = st.sidebar.number_input("Connection Duration", 0, 100000, 0)

    protocol_type = st.sidebar.selectbox(
        "Protocol Type", ["tcp", "udp", "icmp"]
    )

    service = st.sidebar.text_input(
        "Service (e.g. http, ftp, smtp)", "http"
    )

    flag = st.sidebar.text_input(
        "Connection Flag (e.g. SF, REJ)", "SF"
    )

    src_bytes = st.sidebar.number_input("Source Bytes", 0, 1_000_000, 0)
    dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 1_000_000, 0)

    count = st.sidebar.number_input("Access Frequency (count)", 0, 500, 0)
    srv_count = st.sidebar.number_input("Service Access Frequency", 0, 500, 0)

    input_dict = {col: 0 for col in feature_columns}

    input_dict["duration"] = duration
    input_dict["protocol_type"] = protocol_type
    input_dict["service"] = service
    input_dict["flag"] = flag
    input_dict["src_bytes"] = src_bytes
    input_dict["dst_bytes"] = dst_bytes
    input_dict["count"] = count
    input_dict["srv_count"] = srv_count

    return pd.DataFrame([input_dict])


input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 System Overview")
    st.markdown(
        """
        This application demonstrates a **Random Forest based Intrusion Detection System (IDS)**  
        that classifies network traffic as:

        - ✅ **Benign**
        - 🚨 **Malicious**

        The model prioritizes **recall**, ensuring malicious traffic is not missed.
        """
    )

with col2:
    st.subheader("⚙️ Model Details")
    st.markdown(
        f"""
        - **Algorithm:** Random Forest  
        - **Decision Threshold:** {threshold}  
        - **Optimization Goal:** High Recall  
        """
    )

st.markdown("---")
st.subheader("🔍 Live Network Traffic Prediction")

if st.button("🚀 Analyze Traffic"):

    cat_cols = ["protocol_type", "service", "flag"]
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    prob = model.predict_proba(input_df)[0]
    classes = model.classes_

    benign_prob = prob[list(classes).index("Benign")]
    malicious_prob = prob[list(classes).index("Malicious")]

    if malicious_prob >= threshold:
        st.error("🚨 **Malicious Network Traffic Detected**")
        risk = "High Risk"
    else:
        st.success("✅ **Benign Network Traffic Detected**")
        risk = "Low Risk"

    st.markdown(f"### 🧠 Risk Level: **{risk}**")

    st.markdown("### 🔢 Prediction Confidence")
    prob_df = pd.DataFrame(
        {
            "Class": ["Benign", "Malicious"],
            "Probability": [benign_prob, malicious_prob]
        }
    )
    st.bar_chart(prob_df.set_index("Class"))

    st.info(
        f"""
        **Threat Interpretation**
        - Malicious Probability: {malicious_prob:.2%}
        - Decision based on optimized Random Forest model
        """
    )

st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Final Year Project – AI-Based Intrusion Detection System
    </p>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"Model expects {len(feature_columns)} features (NSL-KDD standard)"
)
