from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Disable parallel tokenization to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_name):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def validate_bookmarks(bookmarks):
    if not bookmarks:
        raise ValueError("No bookmarks provided.")
    if not isinstance(bookmarks, list) or not all(isinstance(item, dict) for item in bookmarks):
        raise ValueError(f"Incorrect bookmarks format: {bookmarks}")

def encode_texts(model, texts):
    try:
        return model.encode(texts)
    except Exception as e:
        raise RuntimeError(f"Error encoding texts: {e}")

def perform_clustering(embeddings, distance_threshold=2):
    try:
        link_matrix = linkage(embeddings, method='ward')
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, compute_full_tree=True)
        clusters = clustering.fit_predict(embeddings)
        return clusters, link_matrix
    except Exception as e:
        raise RuntimeError(f"Error during clustering: {e}")

def cluster_texts(bookmarks):
    validate_bookmarks(bookmarks)
    model = load_model('all-MiniLM-L6-v2')
    texts = [f"{bookmark['name']} {bookmark['url']}" for bookmark in bookmarks]
    embeddings = encode_texts(model, texts)
    clusters, link_matrix = perform_clustering(embeddings)
    return clusters, len(set(clusters)), embeddings, link_matrix
