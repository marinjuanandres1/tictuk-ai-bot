import backend
import streamlit as st

if prompt := st.chat_input():

    st.chat_message("User").write(prompt)
    with st.chat_message("Assistant"):
        st.write(" thinking...")
        response = backend.get_answer(prompt)
        st.write(response)