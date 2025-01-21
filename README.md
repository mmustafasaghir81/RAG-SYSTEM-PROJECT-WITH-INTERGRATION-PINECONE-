# 📚 RAG-SYSTEM-PROJECT-WITH-INTERGRATION-PINECONE-Step-by-Step Guide
This project demonstrates how to create a Retrieval-Augmented Generation (RAG) system by integrating Pinecone, LangChain, and Google Gemini. The steps cover initializing Pinecone, embedding documents, and running similarity searches with the help of Google Gemini embeddings.

# Step 1: Install Required Libraries
First, install the necessary libraries for the project.
These include langchain-pinecone and langchain-google-genai for Pinecone and Google Gemini integration.
```bash
%pip install -qU langchain-pinecone langchain-google-genai
```
# Step 2: Set Up Environment Variables
Retrieve and configure the API keys for Pinecone and Google Gemini using Colab's user data or directly as environment variables.
```bash
import os
from google.colab import userdata

# Fetch your API keys from Colab's user data
pinecone_api_key = userdata.get('pinecone')  # Pinecone API key
google_api_key = userdata.get('GOOGLE_API_KEY')  # Google API key

# Set environment variables
os.environ["PINECONE_API_KEY"] = pinecone_api_key  # Required by Pinecone
os.environ["GEMINI_API_KEY"] = google_api_key  # Required by Google Gemini
```
# Step 3: Initialize Pinecone Client
Set up the Pinecone client, create an index if it doesn’t already exist, and connect to it.
```bash
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone()

# Define the index name and specifications
index_name = "rag-project-piaci"

# Create the index if it doesn't already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",  # Use cosine similarity for matching
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)
```
# Step 4: Configure Google Gemini Embeddings
Set up the Google Gemini embeddings to convert text into vector representations.
```bash
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Use the specified embedding model
    google_api_key=google_api_key  # API key for authentication
)

# Test embedding by converting a query into a vector
vector = embeddings.embed_query("we are building a RAG Text")

# Print the first 5 elements of the vector for verification
print(vector[:5])
```
# Step 5: Initialize Pinecone Vector Store
Create a vector store using the Pinecone index and Google Gemini embeddings.
```bash
from langchain_pinecone import PineconeVectorStore

# Initialize Pinecone vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
```
# Step 6: Add Documents to the Vector Store
Prepare a list of documents and embed them into the Pinecone vector store.
```bash
from langchain_core.documents import Document

# Define the documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Add documents to the vector store
vector_store.add_documents(documents=documents)
```
# Step 7: Perform Similarity Search
Run a similarity search using the embedded query against the documents stored in Pinecone.
```bash
# Query the vector store and perform a similarity search
results = vector_store.similarity_search_with_score(
    "What factors are considered when generating recommendations for users?"
)

# Display the results
print("Similarity Search Results:")
for result, score in results:
    print(f"Document: {result.page_content}, Score: {score}")
```
# Step 8: Find the Best Match
Identify the document with the highest similarity score.
```bash
# Find the highest score among the results
highest_score = float('-inf')
best_match = None

for result, score in results:
    if score > highest_score:
        highest_score = score
        best_match = result

# Print the document with the highest score
print("\nBest Match:")
print(f"Document: {best_match.page_content}, Score: {highest_score}")
```


# Conclusion
# The steps above walk through the creation of a RAG system with Pinecone, LangChain, and Google Gemini. This system retrieves the most relevant documents based on the user's query using vector-based similarity search.









