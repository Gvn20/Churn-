import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    model  = tf.keras.models.load_model('model_ann.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

def predict(age, gender, tenure, usage_freq, support_calls,
            payment_delay, subscription, contract, total_spend, last_interaction):
    gender_enc   = 0 if gender == "Female" else 1
    sub_enc      = {"Basic": 0, "Premium": 1, "Standard": 2}[subscription]
    contract_enc = {"Annual": 0, "Monthly": 1, "Quarterly": 2}[contract]
    input_data   = np.array([[age, gender_enc, tenure, usage_freq,
                               support_calls, payment_delay, sub_enc,
                               contract_enc, total_spend, last_interaction]])
    input_scaled = scaler.transform(input_data)
    prob = float(model.predict(input_scaled)[0][0])
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    pred = 1 if prob > 0.5 else 0
    return pred, prob

st.set_page_config(page_title="Customer Churn Predictor", page_icon="🎯")
st.title("🎯 Customer Churn Predictor")
st.markdown("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")
st.divider()

st.subheader("📋 Data Pelanggan")
col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age",                   min_value=18,  max_value=100, value=30)
    gender         = st.selectbox("Gender",                   ["Female", "Male"])
    tenure         = st.number_input("Tenure (bulan)",        min_value=0,   max_value=100, value=12)
    usage_freq     = st.number_input("Usage Frequency",       min_value=0,   max_value=100, value=10)
    support_calls  = st.number_input("Support Calls",         min_value=0,   max_value=50,  value=2)

with col2:
    payment_delay    = st.number_input("Payment Delay (hari)",    min_value=0,   max_value=100, value=5)
    subscription     = st.selectbox("Subscription Type",          ["Basic", "Premium", "Standard"])
    contract         = st.selectbox("Contract Length",            ["Annual", "Monthly", "Quarterly"])
    total_spend      = st.number_input("Total Spend",             min_value=0.0, value=500.0, step=10.0)
    last_interaction = st.number_input("Last Interaction (hari)", min_value=0,   max_value=100, value=10)

st.divider()

if st.button("🔍 Prediksi", use_container_width=True, type="primary"):
    pred, prob = predict(age, gender, tenure, usage_freq, support_calls,
                         payment_delay, subscription, contract, total_spend, last_interaction)

    st.subheader("📊 Hasil Prediksi")
    if pred == 1:
        st.error("⚠️ Pelanggan diprediksi akan **CHURN**")
        col_a, col_b = st.columns(2)
        col_a.metric("Probabilitas Churn",       f"{prob * 100:.2f}%")
        col_b.metric("Probabilitas Tidak Churn", f"{(1-prob) * 100:.2f}%")
    else:
        st.success("✅ Pelanggan diprediksi **TIDAK CHURN**")
        col_a, col_b = st.columns(2)
        col_a.metric("Probabilitas Tidak Churn", f"{(1-prob) * 100:.2f}%")
        col_b.metric("Probabilitas Churn",       f"{prob * 100:.2f}%")

    st.session_state.history.append({
        "Age": age, "Gender": gender, "Tenure": tenure,
        "Usage Freq": usage_freq, "Support Calls": support_calls,
        "Payment Delay": payment_delay, "Subscription": subscription,
        "Contract": contract, "Total Spend": total_spend,
        "Last Interaction": last_interaction,
        "Hasil": "🔴 CHURN" if pred == 1 else "🟢 TIDAK CHURN",
        "Prob Churn": f"{prob * 100:.2f}%",
        "Prob Tidak Churn": f"{(1-prob) * 100:.2f}%"
    })

if len(st.session_state.history) > 0:
    st.divider()
    st.subheader("📜 Riwayat Prediksi")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.caption("Model: ANN/MLP | Dataset: Customer Churn | Framework: TensorFlow/Keras")