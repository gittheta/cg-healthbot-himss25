# Create a simple vector store

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# read in FHIR bundle
documents = SimpleDirectoryReader(input_files=['working/raw_fhir/sarah-brown-bundle-250302.json']).load_data()

# create vector store
index_json = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model
)

# write out vector store
index_json.storage_context.persist("working/vector_store/index_json_sb_250302")
