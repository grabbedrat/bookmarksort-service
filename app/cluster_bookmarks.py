from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_texts(bookmarks):
    texts = [f"{bookmark['name']} {bookmark['url']}" for bookmark in bookmarks]
    embeddings = model.encode(texts)
    num_clusters = 20
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, num_clusters, embeddings

# Sample bookmarks data
bookmarks = [
    {"name": "Example Bookmark 1", "url": "https://example.com"},
    {"name": "Example Bookmark 2", "url": "https://example2.com"}
]
