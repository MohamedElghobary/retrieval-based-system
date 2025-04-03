import cohere
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
import os
import time

from config import COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY

# Download NLTK tokenizer models (only needs to be run once)
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize Cohere API
co = cohere.Client(COHERE_API_KEY)

# Initialize Qdrant client with a custom timeout (in seconds)
qdrant = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

# Verify Qdrant connection before proceeding
try:
    collections = qdrant.get_collections()
    print("Connected to Qdrant! Available collections:", collections)
except Exception as e:
    print("Failed to connect to Qdrant:", e)
    exit()

# 1. Preprocessing Module: Meaningful Chunking
def chunk_text(text):
    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= 150:  # Adjust chunk size as needed
            current_chunk += " " + sentence.strip()
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence.strip()  # Start a new chunk

    if current_chunk:  # Add any remaining text as a final chunk
        chunks.append(current_chunk.strip())

    return chunks

# 2. Embedding Layer: Generate embeddings
def generate_embeddings(chunks):
    return co.embed(texts=chunks, model="embed-english-v2.0").embeddings

# 3. Vector Database: Collection Management
collection_name = "rag_chunks"

# Check if collection exists in Qdrant
collections = qdrant.get_collections().collections
collection_names = [c.name for c in collections]

if collection_name not in collection_names:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(generate_embeddings([""]))[0]),  # Create a dummy embedding for size
        distance=Distance.COSINE
    )

# 4. Indexing: Insert chunks into Qdrant with retry logic
def insert_chunks_into_qdrant(chunks, embeddings, batch_size=100, max_retries=3):
    for i in range(0, len(chunks), batch_size):
        points = [
            PointStruct(id=i + j, vector=emb, payload={"text": chunks[i + j]}) 
            for j, emb in enumerate(embeddings[i:i + batch_size])
        ]
        
        retries = 0
        while retries < max_retries:
            try:
                qdrant.upsert(collection_name=collection_name, points=points)
                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Error during upsert: {e}. Retrying ({retries + 1}/{max_retries})...")
                retries += 1
                time.sleep(1)  # Wait before retrying

# Function to handle file input
def process_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    insert_chunks_into_qdrant(chunks, embeddings)
    print(f"Processed and stored text from {file_path}")

# 5. Retrieval Module: Query Processing
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
    results = qdrant.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)
    return results

# 6. Generation Module: Generate response using Cohere
def generate_response(relevant_chunks, user_query):
    context = " ".join([result.payload['text'] for result in relevant_chunks])
    
    response = co.generate(
        model="command-r-plus",  # Cohere's best RAG model
        prompt=f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {user_query}",
        max_tokens=150,
        temperature=0.3
    )
    
    return response.generations[0].text.strip()

# Main function to handle user input and generate response
def main(user_query):
    relevant_chunks = retrieve_relevant_chunks(user_query)
    response = generate_response(relevant_chunks, user_query)
    return response

# Example usage
if __name__ == "__main__":
    # Process a file input
    file_path = "docs/behaviuor1.txt"  # Replace with the path to your input file
    process_file(file_path)

    # Example query
    user_query = "How Can We Avoid Tantrums?"
    generated_response = main(user_query)
    print("🤖 AI Response:", generated_response)
