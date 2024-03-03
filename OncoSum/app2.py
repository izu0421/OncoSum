import streamlit as st
from transformers import pipeline
summarizer = pipeline("summarization")
st.title('Medical Conversation Summarizer')

# Text area for input
conversation_text = st.text_area("Enter the medical conversation here:", height=300)

if st.button('Summarize Conversation'):
    if conversation_text:
        # Summarize the conversation
        summary = summarizer(conversation_text, max_length=130, min_length=25, do_sample=False)
        
        # Display the summary
        st.subheader('Summary')
        st.write(summary[0]['summary_text'])
    else:
        st.write("Please enter a conversation to summarize.")
