import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext


# Load environment variables
load_dotenv()

# RAG Implementation (Uncomment if needed for document retrieval)
# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_document(documents, show_progress=True)
# query_engine = index.as_query_engine()

class Chain:
    def __init__(self):
        # Initialize ChatGroq with the required API key and model
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile"
        )

    def process_input(self, user_input):
        # Construct a specific prompt with guardrails
        crime_prompt = f"""
        You are tasked with summarizing the following crime narration and strictly identifying only the crime, the relevant Indian Penal Code (IPC) section, and a landmark judgment associated with that crime.
        Ensure that your response is limited to:
        1. The identified crime.
        2. The IPC section relevant to the crime.
        3. A landmark judgment, if applicable.

        If the information is not relevant to crimes, IPC sections, or landmark judgments, ignore it. Do not provide any unrelated information. If you cannot identify a crime or IPC section, explicitly state "Unable to identify a relevant IPC section or landmark judgment."

        Please return the result in the following structured format:
        Crime: [identified crime]\n
        IPC Section: [relevant IPC section]\n
        Landmark Judgment: [related landmark judgment, if applicable]\n

        Narration: "{user_input}"
        """

        # Measure latency
        start_time = time.time()
        response = self.llm.invoke(crime_prompt)
        end_time = time.time()

        # Calculate latency
        latency = end_time - start_time
        return response.content if response and response.content else "No response from the model.", latency


# Streamlit app
def create_streamlit_app(chain):
    st.title("LegalNavi AI Model")

    incident_input = st.text_area("Narrate a crime incident:")
    submit_button = st.button("Submit")

    if submit_button:
        if incident_input.strip():  # Check for non-empty input
            # Get model response and latency
            response, latency = chain.process_input(incident_input)
            st.write("Model Response:")
            st.write(response)

            # Display latency
            st.write(f"Latency: {latency:.4f} seconds")


if __name__ == "__main__":
    chain_instance = Chain()
    create_streamlit_app(chain_instance)
