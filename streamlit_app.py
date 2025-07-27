
import streamlit as st
import requests

st.title("📚 Homework Helper")

question = st.text_area("Enter your homework question:")

if st.button("Get Help"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post("https://homework-helper-backend.com/predict", json={"question": question})
            if response.status_code == 200:
                st.success(response.json()["response"])
            else:
                st.error("Failed to get a response.")
