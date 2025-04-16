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

To keep the Economic Times RAG Agent useful, effective, and scalable, we plan to move forward with several improvements:

- **Additional Data Details:**  
  - **Metadata Enrichment:** Besides the article title and content, we can add more details like publication time, author bio, tags/categories, and even sentiment scores.  
  - **Media and Supplementary Information:** Consider extracting images, infographics, and related links to provide users with richer content alongside the text.

- **Better Text Segmentation:**  
  - **Improved Chunking:** Develop smarter techniques for splitting long articles into meaningful sections rather than fixed-length chunks. This can include paragraph-level segmentation or the use of natural breakpoints (such as headers or sub-headings).  
  - **Dynamic Context Building:** When retrieving snippets for the query, design a system to select not just the top N chunks, but those which best capture the article’s context, possibly merging overlapping sections for coherence.

- **Improved Retrieval Options:**  
  - **Exploring Alternative Vector Engines:** As the dataset grows, it might be beneficial to test managed vector search platforms like Pinecone or Qdrant. These tools often provide additional features like real-time scaling and high-performance search capabilities.  
  - **Hybrid Search Techniques:** Integrate classical keyword-based search methods along with vector search to boost precision, especially for queries that need exact phrases.

- **User-Friendly Interface:**  
  - **Web-Based Front End:** Develop a lightweight, responsive web interface using frameworks like React or Vue.js. This would provide a more engaging user experience beyond the command-line interface.  
  - **Mobile Application:** Consider creating a mobile-friendly version of the interface to reach users on the go.  
  - **Interactive Visualizations:** Add features like highlighted text in articles, clickable results for deeper dives, and real-time feedback on search progress.

- **Model Tuning:**  
  - **Domain-Specific Fine-Tuning:** Fine-tune the groq language model with Economic Times-specific content. This could involve training on historical articles and curated Q&A pairs to improve relevance and accuracy.  
  - **Continuous Learning:** Implement mechanisms for the system to learn from user interactions, which can guide periodic updates to the model and retrieval algorithms.

