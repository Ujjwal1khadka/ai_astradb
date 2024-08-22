import os  
import streamlit as st 
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.embeddings import SentenceTransformerEmbeddings  
from langchain.vectorstores import Cassandra as LangchainCassandra  # Use Cassandra for AstraDB
from langchain.llms import OpenAI  
from langchain.chains import RetrievalQA 

# Load API keys and AstraDB credentials from environment variables
astra_db_client_id = os.getenv("ASTRA_DB_CLIENT_ID")
astra_db_client_secret = os.getenv("ASTRA_DB_CLIENT_SECRET")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize AstraDB connection
cloud_config = {
    'secure_connect_bundle': 'path_to_secure_connect_bundle.zip'  # Replace with your secure connect bundle path
}
auth_provider = PlainTextAuthProvider(astra_db_client_id, astra_db_client_secret)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# Initialize OpenAI with GPT-4o Mini
llm = OpenAI(model="gpt-4o-mini", temperature=0.7)  

# Create a vector store with embeddings using AstraDB
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Load the embedding model
vectorstore = LangchainCassandra(session, embeddings, keyspace="your_keyspace_name")  # Specify your keyspace

# Set up the RetrievalQA chain
qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())  # Create the QA chain

# Streamlit UI
st.title("VitafyAI")  # Set the title of the web app
user_input = st.text_input("Ask a question about Vitafy products:")  # Input field for user questions

if st.button("Submit"):  # Button to submit the question
    if user_input:  # Check if user input is not empty
        try:
            # Get the answer from the QA chain
            answer = qa_chain.run(user_input)  # Run the QA chain with the user input
            st.write(answer)  # Display the answer to the user
        except Exception as e:  # Handle any errors that occur
            st.error(f"An error occurred: {e}")  # Display an error message
    else:
        st.warning("Please enter a question.")  # Prompt user to enter a question if input is empty