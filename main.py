import cohere
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY


# Initialize Cohere API
co = cohere.Client(COHERE_API_KEY)

# Initialize Qdrant client using API mode
qdrant = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# Verify Qdrant connection before proceeding
try:
    collections = qdrant.get_collections()
    print("Connected to Qdrant! Available collections:", collections)
except Exception as e:
    print("Failed to connect to Qdrant:", e)
    exit()

# Sample long text
text = """
Machine learning is a field of artificial intelligence (AI) that allows computers to learn from data without being explicitly programmed.
Deep learning is a subset of machine learning that uses neural networks with many layers.
One of the most common applications of deep learning is natural language processing (NLP).
NLP enables machines to understand, interpret, and generate human language.
"""

# 1. Preprocessing Module: Chunking
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    return splitter.split_text(text)

chunks = chunk_text(text)

# 2. Embedding Layer: Generate embeddings
def generate_embeddings(chunks):
    return co.embed(texts=chunks, model="embed-english-v2.0").embeddings

embeddings = generate_embeddings(chunks)

# 3. Vector Database: Collection Management
collection_name = "rag_chunks"

# Check if collection exists in Qdrant
collections = qdrant.get_collections().collections
collection_names = [c.name for c in collections]

if collection_name not in collection_names:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )

# 4. Indexing: Insert chunks into Qdrant
def insert_chunks_into_qdrant(chunks, embeddings):
    points = [
        PointStruct(id=i, vector=emb, payload={"text": chunks[i]}) for i, emb in enumerate(embeddings)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)

insert_chunks_into_qdrant(chunks, embeddings)

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
    user_query = "What is NLP?"
    generated_response = main(user_query)
    print("🤖 AI Response:", generated_response)