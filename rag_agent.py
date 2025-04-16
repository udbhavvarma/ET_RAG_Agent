import openai
import json
import faiss
import numpy as np
import groq
from sentence_transformers import SentenceTransformer
import os
import dotenv

dotenv.load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

DATA_FILE = "articles.json"
INDEX_FILE = "faiss.index"


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_articles():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def load_faiss_index():
    return faiss.read_index(INDEX_FILE)

def query_index(query, index, articles, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = []
    for idx in indices[0]:
        if idx < len(articles):
            results.append(articles[idx])
    return results

def generate_answer(query, context):
    prompt = (
        f"Using the following excerpts from Economic Times articles:\n{context}\n\n"
        f"Answer the following question:\n{query}"
    )
    response = groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        api_key=groq_api_key
    )
    return response.choices[0].message.content

def main():
    query = input("Enter your query: ")
    articles = load_articles()
    index = load_faiss_index()
    retrieved_articles = query_index(query, index, articles, top_k=3)
    context = "\n\n".join(
        [f"Title: {art['title']}\nContent: {art['content'][:500]}..." for art in retrieved_articles]
    )
    
    answer = generate_answer(query, context)
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
