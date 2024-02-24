from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_texts(bookmarks):
    """
    Clusters bookmarks based on the combined information of their titles and URLs.
    Future modifications may include tags with light weighting.
    
    Input:
    - bookmarks: A list of dictionaries, where each dictionary represents a bookmark
      and must have at least 'name' and 'url' keys.
      
    Output:
    - clusters: A list of cluster labels indicating the cluster each bookmark belongs to.
    - num_clusters: The number of clusters formed.
    """
    
    # Combine title and URL into a single string for each bookmark
    texts = [f"{bookmark['name']} {bookmark['url']}" for bookmark in bookmarks]
    
    # Encode the combined texts to get their embeddings
    embeddings = model.encode(texts)
    
    # Determine the number of clusters, ensuring there's at least one and no more than 8
    num_clusters = min(8, len(set(texts)))
    
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters, num_clusters
