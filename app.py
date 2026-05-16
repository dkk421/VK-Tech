import streamlit as st

st.title("Medical AI Assistant")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    st.success("Dataset uploaded")