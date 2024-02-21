from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_texts(texts):
    embeddings = model.encode(texts)
    num_clusters = min(8, len(set(texts)))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, num_clusters
