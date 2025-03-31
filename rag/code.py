import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import gradio as gr

# Set API key
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

# Load dat
data_dir = "repo_fetching/Repositories_path/awesome-cheatsheets/"
documents = SimpleDirectoryReader(data_dir).load_data()

# Initialize embedding model
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.environ["GOOGLE_API_KEY"])

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_project")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=embed_model,
    show_progress=True
)

# Initialize LLM and query engine
llm = Gemini(model_name="models/gemini-1.5-pro", api_key=os.environ["GOOGLE_API_KEY"])
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

# Gradio interface
def answer_query(query):
    response = query_engine.query(query)
    return response.response

interface = gr.Interface(fn=answer_query, inputs="text", outputs="text", title="RAG with Gemini 1.5 and ChromaDB")
interface.launch()