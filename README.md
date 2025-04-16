# Economic Times RAG Agent

## Purpose

The Economic Times RAG Agent is built to help you quickly find and understand the key details buried within Economic Times articles. Many users get overwhelmed by long articles, struggle to locate important facts, or waste time digging for answers. This project uses web scraping, a vector search engine, and a language model powered by groq to answer your questions with clear, relevant information. The goal is to ease common issues like information overload, missed insights, and time-consuming research.

## Business Value

- **Better Access to Information:**  
  Quickly locate important facts and trends from Economic Times articles without reading every word.

- **Time Saving:**  
  Automate the process of searching and summarizing lengthy articles, freeing you to focus on making decisions.

- **Improved Decision Making:**  
  Provide clear answers that combine data from multiple sources, so you can be confident in your choices.

- **Scalable and Adaptable:**  
  The system is designed to grow and can be easily updated with new data sources or features.

## Technical Explanation

### Architecture Overview

1. **Data Scraping & Processing**  
   - **Scraper (`et_scraper.py`):**  
     Uses BeautifulSoup to fetch Economic Times articles. It extracts details like titles, content, and other useful info.  
   - **Daily Updates via CRON:**  
     A scheduled CRON job runs the scraper each day to keep the data fresh.
   - **Persistent Storage:**  
     Articles are saved as JSON files, and their embeddings are indexed with FAISS for fast search.

2. **Embeddings and Indexing**  
   - **Text Embedding:**  
     The `SentenceTransformer` (model: `all-MiniLM-L6-v2`) converts article text into numeric embeddings.
   - **Vector Search with FAISS:**  
     These embeddings are stored in FAISS to quickly find the articles that best match your query.

3. **Retrieval-Augmented Generation (RAG)**  
   - **RAG Agent (`rag_agent.py`):**  
     When you ask a question, this script retrieves the most relevant articles and uses a groq-powered language model to generate a clear answer.
   - **Chat Interface (`chat_interface.py`):**  
     Provides a friendly chat-like interface for entering questions and receiving responses based on the retrieved content.


### Dependencies

- Python 3.x
- `requests`
- `beautifulsoup4`
- `sentence-transformers`
- `faiss-cpu`
- `groq`
- `numpy`
- `json`

## Future Steps

- **Add More Article Details:**  
  We can include more information such as publication dates, author bios, categories, and even images or links with each article.

- **Improve Text Segmentation:**  
  We can work on smarter ways to split long articles into smaller, meaningful sections. This will help us give better context for each query.

- **Enhance the Search Function:**  
  We can try out other vector search tools like Pinecone or Qdrant as our dataset grows. We might also combine keyword and vector searches for more precise results.

- **Build a Better Interface:**  
  We can create a user-friendly web interface using frameworks like React or Vue.js. We could even build a mobile-friendly version to make our tool more accessible.

- **Tune the Language Model:**  
  We can fine-tune the groq language model using Economic Times content. We can also let the system learn from user feedback and improve over time.

- **Improve Monitoring and Feedback:**  
  We can set up simple monitoring tools to track how our system performs. We can also gather user feedback to fix issues and enhance the overall experience.
