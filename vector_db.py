from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
from collections import OrderedDict

# Load environment variables from .env file
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load the pdf
loader = PyPDFLoader("./data/BNS.pdf")
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text = text_splitter.split_documents(documents)

# Text Embedding (Ensure you're using the correct Hugging Face model)
embedding_model = HuggingFaceEmbeddings(model_name="meta-llama/Llama-2-7b")

try:
    embeddings = embedding_model.embed_documents([doc.page_content for doc in text])
    print("Vector Embeddings Created Successfully")
except Exception as e:
    print(f"Error creating vector embeddings: {e}")

# Initializing Chroma Vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory="data")

# Add documents to vector store
vector_store.add_documents(documents=text)

# Validate the setup
try:
    text_query = "What landmark judgment is for Murder?"
    results = vector_store.similarity_search(query=text_query)

    unique_results = OrderedDict()
    for doc in results:
        if doc.page_content not in unique_results:
            unique_results[doc.page_content] = doc

    final_results = list(unique_results.values())[:3]
    print(f"Unique query results: {final_results}")

except Exception as e:
    print(f"Error during query: {e}")
