import os
import logging

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini


load_dotenv()
GOOGLE_API_KEY =os.getenv("GOOGLE_API_KEY")


# with st.sidebar:
#     st.title("'🤗💬 Chat with your Data'")
#     st.markdown('''
  
#                 ''')
def main():
    # st.header("Capgemini Health AI")
    st.title("Capgemini Health AI")

    # reader = SimpleDirectoryReader(input_dir="..\\working\\raw_fhir\\")
    # docs = reader.load_data()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    llm = Gemini(model="models/gemini-2.0-flash",
                 system_prompt="You are a medical expert who can answer questions on a patient's health history and provide analysis based on it.")
    
    logging.log(logging.INFO, 'loading index...')
    
    storage_context = StorageContext.from_defaults(persist_dir="..\\working\\vector_store\\index_json")
    
    index = load_index_from_storage(
        storage_context,
        # we can optionally override the embed_model here
        # it's important to use the same embed_model as the one used to build the index
        embed_model=embed_model
    )

    query=st.text_input("Ask questions related to your Data")

    if query:
        query_engine = index.as_query_engine(response_mode="compact", llm=llm, similarity_top_k=30, streaming=True)
        response = query_engine.query(query)
        st.write_stream(response.response_gen)

if __name__=='__main__':
    main()    
# ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the ML"))





# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What would you like to know about Sarah?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append({"role": "assistant", "content": response})
