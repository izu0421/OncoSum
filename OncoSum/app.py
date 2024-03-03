import streamlit as st
#from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

import pandas as pd

from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import torch
from rouge import Rouge

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

import numpy as np

openai_api_key = "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"

logo_path = "logo2.png"

### Define prompt
prompt_template_clindoc = """You are a breast oncologist and you have to write a detailed and clear summary\
to document down the conversation you had with the patient.

This letter should be written from the breast oncologist point of view in a first person perspective using "I" and "me" \ 

The conversation history between you as a breast oncologist and your patient is provided below:\
"{text}" \

The clinical documentation should be in the following format\
    Chief Complaints: [Include cancer type and the reason for the consultation] \
    Status: -||- \ 
    Preferred Language: -||-\
    History of presenting illnesses: -||-\
    Past Medical History: -||-\
    Past Surgical History: -||- \
    GYN history: -||-\
    Tobacco use: -||-\
    Social History: -||-\
    Family History:-||-\            
    Physical Examination: -||-\
    Assessment: [Include the cancer type and all mutations, overall goals of the care, detailed treatment options, risk and benefits of the treatment options and the detailed follow-up plan for the next visit. This section should be as detailed as possible.] \
    Plan: [This section should be kept short. Include everything that patient should take note of and follow-up. Output the results in bullet points] \

Stop after "Plan"

If any of the sections listed cannot be found in the conversation history, do not make up any information. \ 

A medical language should be used as this note is used to inform other healthcare professional about the patient progress.\

CLINICAL SUMMARY NOTE:"""

prompt_template_pat = """You are a breast oncologist and you have to write a concise and clear summary\
to your breast cancer patients in layman language on the items discussed during the conversations\
The conversation history between you as a breast oncologist and your patient is provided below:
"{text}"
This letter should be written from the point of view of the patient in a first person perspective using "I" and "me" \
Display the result in a markdown format: \
    Purpose of the visit: [Include the reason for the visit]
    Summary of the discussion: [Include overall goals of the care, detailed treatment options, risk and benefits of the treatment options and the detailed follow-up plan for the next visit. Output the result in bulletpoints]
    Next Steps: [Include everything that patient should take note of and follow-up. Output the results in bullet points]
PATIENT SUMMARY NOTE:"""

# Page title
st.set_page_config(page_title='OncoSum')
st.image(logo_path, width=200)  # Adjust the width as necessary to fit your UI design

st.title('OncoSum')

# sidebar boxes
model_selected = st.sidebar.selectbox(
    "Model selection:",
    ("GPT 3.5 Turbo", "OncoSum"), index=None, 
    placeholder="Select model..."
)

summary_selected = st.sidebar.selectbox(
    "Types of summary:",
    ("Clinical Documents", "Patient Note"), index=None,
    placeholder="Select summary..."
)

if summary_selected == "Clinical Documents":
    prompt = prompt_template_clindoc
elif summary_selected == "Patient Note":
    prompt = prompt_template_pat

if model_selected == "GPT 3.5 Turbo":
    model_name_full = "gpt-3.5-turbo-0125"
elif model_selected == "Fine tuned llamma (OncoSum)":
    model_name_full = "gpt-4-0125-preview"

model_name_full = "gpt-3.5-turbo-0125"

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=model_name_full)

###LOAD CONVOS
pdf_loader = PyPDFLoader("Synthetic_Conversations.pdf")
pages = pdf_loader.load()
output_parser = StrOutputParser()

def generate_response(llm, pages, prompt):
    # Text summarization
    clindoc_prompt = ChatPromptTemplate.from_template(prompt)
    naive_chain = clindoc_prompt | llm | output_parser

    return naive_chain.invoke({"text": pages})

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    st.form_submit_button('Upload pdf')
    submitted = st.form_submit_button('Generate')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            naive_sum = generate_response(llm, pages, prompt)
            result.append(naive_sum)
            del openai_api_key

if len(result):
    st.info(naive_sum)


