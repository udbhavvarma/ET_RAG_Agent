import requests
import json
import os
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


DATA_FILE = "articles.json"   
INDEX_FILE = "faiss.index"      
META_FILE = "metadata.json"     


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_existing_data():
    """Load existing articles from disk."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_articles(articles):
    """Save the list of articles to disk."""
    with open(DATA_FILE, "w") as f:
        json.dump(articles, f, indent=2)

def load_faiss_index(dim):
    """Load existing FAISS index or create a new one."""
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    
    return faiss.IndexFlatL2(dim)

def save_faiss_index(index):
    """Save the FAISS index to disk."""
    faiss.write_index(index, INDEX_FILE)

def get_article_links():
    """
    Fetch article links from a target ET page.
    Modify the URL and parsing logic as needed for your target section.
    """
    url = "https://economictimes.indiatimes.com/industry"  
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        
        if "/articleshow/" in href:
            full_link = ("https://economictimes.indiatimes.com" + href) if href.startswith("/") else href
            links.append(full_link)
    
    return list(set(links))

def scrape_article(url):
    """
    Scrape an article page.
    Adjust the selectors based on ET's website structure.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        title = soup.find("h1").get_text(strip=True)
    except Exception:
        title = "No Title Found"
    
    content_div = soup.find("div", class_="Normal")
    if content_div:
        content = content_div.get_text(separator=" ", strip=True)
    else:
        content = soup.get_text(separator=" ", strip=True)
    article = {
        "url": url,
        "title": title,
        "content": content
    }
    return article

def update_index(articles):
    """Generate embeddings for articles and build a FAISS index."""
    texts = [article["content"] for article in articles]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

def main():
    print("Starting Economic Times scraping job...")
    existing_articles = load_existing_data()
    existing_urls = set(article["url"] for article in existing_articles)
    
    print("Fetching article links...")
    links = get_article_links()
    print(f"Found {len(links)} article links.")
    
    new_articles = []
    for link in links:
        if link not in existing_urls:
            try:
                print(f"Scraping article: {link}")
                article = scrape_article(link)
                new_articles.append(article)
            except Exception as e:
                print(f"Error scraping {link}: {e}")
    
    if new_articles:
        print(f"Scraped {len(new_articles)} new articles.")
        all_articles = existing_articles + new_articles
        save_articles(all_articles)
        
        index = update_index(all_articles)
        save_faiss_index(index)
        print("FAISS index updated successfully.")
    else:
        print("No new articles found.")

if __name__ == "__main__":
    main()
