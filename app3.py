import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Document-Based Question Answering System")

uploaded_files = st.file_uploader(
    "Choose PDF or CSV files", type=["pdf", "csv"], accept_multiple_files=True
)

if uploaded_files:
    files_to_send = [("files", (file.name, file.getvalue(), "application/octet-stream")) for file in uploaded_files]

    with st.spinner("Uploading and processing..."):
        response = requests.post(f"{API_URL}/upload/", files=files_to_send)

    if response.status_code == 200:
        st.success("Files processed successfully!")
    else:
        st.error("Error uploading files.")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the uploaded documents")

if query:
    st.chat_message("user").write(query)

    with st.spinner("Thinking..."):
        response = requests.post(f"{API_URL}/query/", json={"query": query})
    
    if response.status_code == 200:
        answer = response.json()["response"]
        st.chat_message("assistant").write(answer)
    else:
        st.error("Error fetching response.")
