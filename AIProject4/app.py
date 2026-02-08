import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("ids_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
metrics = pickle.load(open("metrics.pkl", "rb"))

st.set_page_config(
    page_title="AI-Based Intrusion Detection System",
    page_icon="🔐",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center;'>🔐 AI-Based Intrusion Detection System</h1>
    <p style='text-align:center; font-size:18px;'>
    Supervised Machine Learning System for Network Traffic Classification
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
    **Model:** Decision Tree Classifier  
    **Dataset:** NSL-KDD  
    **Classes:** Benign / Malicious  
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Enter Network Parameters")

def user_input_features():
    duration = st.sidebar.number_input("Connection Duration", 0, 100000, 0)

    protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
    protocol_label = st.sidebar.selectbox("Protocol Type", list(protocol_map.keys()))
    protocol_type = protocol_map[protocol_label]

    service = st.sidebar.number_input(
        "Service Type (Encoded)", 0, 70, 0,
        help="Encoded service as per NSL-KDD dataset"
    )

    flag = st.sidebar.number_input(
        "Connection Flag (Encoded)", 0, 10, 0,
        help="Encoded TCP status flag"
    )

    src_bytes = st.sidebar.number_input("Source Bytes", 0, 1_000_000, 0)
    dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 1_000_000, 0)

    count = st.sidebar.number_input(
        "Access Frequency (Count)", 0, 500, 0,
        help="Connections to same host"
    )

    srv_count = st.sidebar.number_input(
        "Service Access Frequency", 0, 500, 0,
        help="Connections to same service"
    )

    features = [
        duration, protocol_type, service, flag,
        src_bytes, dst_bytes,
        *[0] * 16,    
        count, srv_count,
        *[0] * 17   
    ]

    return np.array([features])


input_data = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 System Overview")
    st.markdown(
        """
        This application demonstrates an **AI-based Intrusion Detection System (IDS)**  
        that analyzes network traffic and classifies it as:

        - ✅ **Benign (Normal Traffic)**
        - 🚨 **Malicious (Potential Intrusion)**

        The system is trained using historical network traffic data
        and can assist in early cyber threat detection.
        """
    )

with col2:
    st.subheader("⚙️ Model Details")
    st.markdown(
        """
        - **Algorithm:** Decision Tree  
        - **Learning Type:** Supervised  
        - **Input Features:** Network traffic parameters  
        - **Output:** Benign / Malicious  
        """
    )

st.markdown("---")
st.subheader("🔍 Live Network Traffic Prediction")
if st.button("🚀 Analyze Traffic"):
    assert input_data.shape[1] == scaler.n_features_in_, (
        f"Expected {scaler.n_features_in_} features, "
        f"but received {input_data.shape[1]}"
    )

    input_scaled = scaler.transform(input_data)

    proba = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    benign_prob = proba[list(classes).index("Benign")]
    malicious_prob = proba[list(classes).index("Malicious")]

    if malicious_prob >= 0.6:
        st.error("🚨 **Malicious Network Traffic Detected**")
        risk = "High Risk"
    elif malicious_prob >= 0.4:
        st.warning("⚠️ **Suspicious Network Traffic Detected**")
        risk = "Medium Risk"
    else:
        st.success("✅ **Benign Network Traffic**")
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
        - Decision based on trained Decision Tree model
        """
    )

st.markdown("---")
st.subheader("📈 Model Evaluation (Test Dataset)")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col1.metric("Precision", f"{metrics['precision']:.2f}")
col2.metric("Recall", f"{metrics['recall']:.2f}")
col2.metric("F1 Score", f"{metrics['f1']:.2f}")

st.subheader("🧩 Confusion Matrix")
cm_df = pd.DataFrame(
    metrics["confusion_matrix"],
    columns=["Predicted Benign", "Predicted Malicious"],
    index=["Actual Benign", "Actual Malicious"]
)
st.dataframe(cm_df)

with st.expander("ℹ️ Feature Explanation"):
    st.markdown(
        """
        - **Connection Duration:** Length of the network session  
        - **Protocol Type:** Communication protocol (TCP / UDP / ICMP)  
        - **Source & Destination Bytes:** Data transferred during the session  
        - **Access Frequency:** Number of connections to same host/service  
        - **Other Features:** Auto-filled to preserve NSL-KDD compatibility  
        """
    )

with st.expander("🛠️ How the System Works"):
    st.markdown(
        """
        1. Network traffic parameters are collected  
        2. Input features are scaled using a trained scaler  
        3. Decision Tree analyzes learned traffic patterns  
        4. Traffic is classified as Benign or Malicious  
        5. Risk level is generated based on prediction confidence  
        """
    )
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Machine Learning Mini Project – Cybersecurity (Intrusion Detection System)
    </p>
    """,
    unsafe_allow_html=True
)
st.caption(
    f"Model expects {scaler.n_features_in_} features (NSL-KDD standard)"
)
st.caption(
    "Predictions are influenced by the full feature space learned during training."
)
# NOTE:
# Full NSL-KDD feature vector requires 41 features.
# Non-user-input features are auto-filled with neutral values
# to maintain compatibility with the trained model.

