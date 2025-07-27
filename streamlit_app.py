import streamlit as st
import joblib
from keras.models import load_model
import numpy as np

# Load models from the same directory (no 'backend/' prefix)
xgb_model = joblib.load("xgboost_model.pkl")
rnn_model = load_model("rnn_model.h5")

st.title("📚 Homework Helper")

# Function to preprocess question (simple version)
def preprocess(question):
    # Example dummy vector for XGBoost (replace with real preprocessing)
    xgb_input = np.array([[len(question), sum(c.isdigit() for c in question)]])
    # Example dummy sequence for RNN (replace with real tokenizer & sequence logic)
    rnn_input = np.random.rand(1, 10, 1)  # Replace with actual preprocessed input
    return xgb_input, rnn_input

question = st.text_area("Enter your homework question:")

if st.button("Get Help"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            xgb_input, rnn_input = preprocess(question)

            # Predict with both models (dummy predictions)
            xgb_result = xgb_model.predict(xgb_input)
            rnn_result = rnn_model.predict(rnn_input)

            # Combine the results (mock example)
            combined_response = f"XGBoost Score: {xgb_result[0]:.2f}, RNN Score: {rnn_result[0][0]:.2f}"

            st.success(f"🤖 Response: {combined_response}")
