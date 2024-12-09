import streamlit as st

try:
    st.write("OPENAI_API_KEY:", st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found. Please configure it in Streamlit Secrets.")
