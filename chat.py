import groq
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import dotenv

dotenv.load_dotenv()

client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# File paths for articles and the FAISS index
DATA_FILE = "articles.json"
INDEX_FILE = "faiss.index"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_articles():
    """Load the stored Economic Times articles."""
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def load_faiss_index():
    """Load the FAISS index from disk."""
    return faiss.read_index(INDEX_FILE)

def query_index(query, index, articles, top_k=3):
    """
    Encode the query, search the FAISS index, and return
    the top_k articles that are most relevant.
    """
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(articles):
            results.append(articles[idx])
    return results

def generate_answer(query, context):
    """
    Create a prompt that incorporates the retrieved context (excerpts
    from Economic Times articles) and the query, and generate an answer using OpenAI.
    """
    prompt = (
        f"Using the following excerpts from Economic Times articles:\n{context}\n\n"
        f"Answer the following question:\n{query}"
    )
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def main():
    articles = load_articles()
    index = load_faiss_index()
    print("Welcome to the Economic Times Chat Agent.")
    print("Type 'exit' or 'quit' to end the chat.\n")
    
    while True:
        query = input("You: ")
        if query.strip().lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break
        
        retrieved_articles = query_index(query, index, articles, top_k=3)
        context = "\n\n".join(
            [f"Title: {art['title']}\nContent: {art['content'][:500]}..." for art in retrieved_articles]
        )
        
        answer = generate_answer(query, context)
        print(f"Agent: {answer}\n")

if __name__ == "__main__":
    main()
