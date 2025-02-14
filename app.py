import streamlit as st
import requests

st.title("PDF Chatbot with NLP")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/upload/", files=files)
    data = response.json()
    st.success(data["message"])
    st.write("Extracted Named Entities:", data["entities"])

query = st.text_input("Ask a question:")
if query:
    response = requests.post("http://127.0.0.1:8000/chat/", json={"query": query})
    st.write("Chatbot:", response.json()["response"])
