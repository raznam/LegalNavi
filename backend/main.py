import os
import shutil
import time
import gc
from typing import List
from contextlib import contextmanager

# Lifespan-like resource setup (manual)
def setup_environment():
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    gc.collect()

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_pdfs(files: List[str]):
    """
    files: list of file paths (PDFs)
    """
    uploaded_files = []

    for filepath in files:
        filename = os.path.basename(filepath)
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Only PDF files. Invalid file: {filename}")
        
        dest_path = os.path.join(UPLOAD_DIR, filename)
        try:
            with open(filepath, "rb") as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            uploaded_files.append(filename)
        except Exception as e:
            raise RuntimeError(f"Failed to upload {filename}: {str(e)}")

    try:
        from embed_doc import embed_documents
        embed_documents()
        return {
            "message": f"{len(uploaded_files)} PDF files uploaded and embedded.",
            "files": uploaded_files
        }
    except Exception as e:
        # Clean up uploaded files if embedding fails
        for filename in uploaded_files:
            path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(path):
                os.remove(path)
        raise RuntimeError(f"Embedding failed: {str(e)}")


def query_documents_cli(query: str):
    try:
        from query_doc import query_documents
        start_time = time.time()
        response = query_documents(query)
        end_time = time.time()

        latency = round(end_time - start_time, 3)
        return {
            "response": response,
            "latency_seconds": latency
        }
    except Exception as e:
        raise RuntimeError(f"Query failed: {str(e)}")



# Example usage (optional):
if __name__ == "__main__":
    setup_environment()
    # Example: upload_pdfs(["./sample1.pdf", "./sample2.pdf"])
    # Example: print(query_documents_cli("Explain IPC 302"))
    # Example: print(delete_data())
