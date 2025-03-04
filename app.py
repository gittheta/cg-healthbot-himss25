import os
import logging
import json
import requests

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage


load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FHIR_SERVER_KEY = os.getenv("FHIR_SERVER_KEY")
FHIR_BASE_URL = 'https://sof-services.hike.health:8000'

avatar_images = {'user':':material/stethoscope:', 'assistant':'bot_avatar.jpeg'}


def get_query_context(provider_specialty):
    r = requests.get(f'{FHIR_BASE_URL}/Patient/ae7eb92d-0ecc-40e4-8e88-91167336a2c6/$everything?_count=400',
                    headers={'Authorization':FHIR_SERVER_KEY})

    print('getting data from FHIR server...')

    if r.status_code == 200:
        patient_data = str(r.json())
    else:
        print(f'Retrieval from FHIR server failed - {r.status_code}, {r.content}. Using local file.')    
        with open('sarah-brown-bundle-250302.json') as fin:
            patient_data = fin.read()

    specialty_focus = f'I\'m a {provider_specialty}. Provide answers in terminology I would use. Only include details relevant to my specialty.'

    initial = f'''Use the following set of patient health data when responding to subsequent queries. 
        {specialty_focus} Health Data: {patient_data}'''
    
    return initial


with st.sidebar:
    cg_logo_url = 'https://www.capgemini.com/wp-content/themes/capgemini2020/assets/images/logo.svg'

    st.markdown(
        f"""
        <div style='display: flex; align-items: center;'>
            <img src='{cg_logo_url}' style='width: 200px'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Health History Assistant")
    st.markdown('''This AI-powered assistant leverages the latest interoperability standards
                to produce tailored, natural language summaries of patient data. With seamless
                integration into the EHR, it enables practitioners 
                to ask questions and gather relevant information for their intake process,
                all while staying within their clinical workflow.''')
    
    grounding_method = st.selectbox('Pick a grounding method', ['Long Context', 'RAG'])

    provider_specialty = st.selectbox('Pick a provider specialty', ['Cardiologist', 'PCP',
                                                                    'Endocrinologist', 'Podiatrist',
                                                                    'Nutritionist', 'Psychiatrist'])

    if st.button('Reset chat'):
        del st.session_state['messages']   


def main():
    # st.title("Health History Assistant ")

    # reader = SimpleDirectoryReader(input_dir="..\\working\\raw_fhir\\")
    # docs = reader.load_data()
    # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    if "llm" not in st.session_state:
        st.session_state.llm = Gemini(model="models/gemini-2.0-flash",
                 system_prompt="You are a knowledgable and helpful AI health history assistant. You can answer questions on a patient's health history and can provide analysis based on it. When you are asked to provide a summary of the patient's health history, the summary should consist of 3 parts - a synopsis describing the main problems of the patient, the patient's treatment history thus far, and recommended follow up treatment the patient should get.")
        print('created llm')
    # logging.log(logging.INFO, 'loading index...')
    
    # storage_context = StorageContext.from_defaults(persist_dir="./index_json")
    
    # index = load_index_from_storage(
    #     storage_context,
    #     # we can optionally override the embed_model here
    #     # it's important to use the same embed_model as the one used to build the index
    #     embed_model=embed_model
    # )

    # # if "messages" not in st.session_state:
    # #     st.session_state.messages = []

    # with open('..\\working\\raw_fhir\\sarah-williams-bundle-250220.json') as fin:
    #     sw_data_raw = fin.read()

    # voice_common = 'Use the voice of a lay person when responding to queries. Use common verbiage.'
    # voice_clinician = 'Use the voice of a clinician when responding to queries. Use clinical verbiage.' 
    # initial = f'Use the following set of patient health data when responding to subsequent queries. {voice_clinician} Health Data: {sw_data_raw}'
    
    if "messages" not in st.session_state:
        query_context = get_query_context(provider_specialty)
        summary_prompt = 'Please provide a summary of the patient\'s health history in a few paragraphs.'
        st.session_state.messages = [ChatMessage(role='user', content=f'{query_context} {summary_prompt}')]
        sum_res = st.session_state.llm.chat(messages=st.session_state.messages)
        st.session_state.messages.append(sum_res.message)


    for message in st.session_state.messages[1:]:
        with st.chat_message(message.role.value, avatar=avatar_images[message.role.value]):
            st.markdown(message.content)

    if prompt := st.chat_input("What would you like to know about Sarah?"):
        with st.chat_message('user', avatar=avatar_images['user']):
            st.markdown(prompt)
        
        st.session_state.messages.append(ChatMessage(role='user', content=prompt))
        response = st.session_state.llm.chat(messages=st.session_state.messages)
 
        with st.chat_message("assistant", avatar=avatar_images['assistant']):
            st.markdown(response.message.content)
            # for r in response:
            #     st.write(r.delta)
        st.session_state.messages.append(response.message)


    # query=st.text_input("Ask questions related to your Data")

    # if query:
    #     query_engine = index.as_query_engine(response_mode="compact", llm=llm, similarity_top_k=30, streaming=True)
    #     response = query_engine.query(query)
    #     st.write_stream(response.response_gen)

if __name__=='__main__':
    main()
