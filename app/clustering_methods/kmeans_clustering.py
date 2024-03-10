from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

# Disable parallel tokenization to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def cluster_texts(bookmarks):
    if not bookmarks:
        print("No bookmarks provided.")
        return [], 0, []

    try:
        # Load the model
        model = load_model('all-MiniLM-L6-v2')
        if model is None:
            return [], 0, []

        if not isinstance(bookmarks, list) or not all(isinstance(item, dict) for item in bookmarks):
            print(f"Incorrect bookmarks format: {bookmarks}")
            return [], 0, []
        
        texts = [f"{bookmark['name']} {bookmark['url']}" for bookmark in bookmarks]

        # Encode texts
        try:
            embeddings = model.encode(texts)
        except Exception as e:
            print(f"Error encoding texts: {e}")
            return [], 0, []

        num_clusters = 20
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        except Exception as e:
            print(f"Error during clustering: {e}")
            return [], 0, []

        return clusters, num_clusters, embeddings

    except Exception as e:
        print(f"Unexpected error: {e}")
        return [], 0, []
