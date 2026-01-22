import  streamlit as st
from rag_chain import answer_question

st.title ("I am your Ai assistant")

user_input = st.text_input ("Ask a question about python:")

if st.button ("Ask"):
    if user_input.strip() !=" ":
        with st.spinner ("Thinking..."):
            response =answer_question(user_input)
            st.write ("###  Answer:")
            st.write (response)
