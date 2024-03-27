import numpy as np
from app.clustering_methods.heirarchical_clustering import cluster_texts

def test_cluster_texts():
    bookmarks = [
        {"name": "Test Bookmark 1", "url": "http://example.com/1"},
        {"name": "Test Bookmark 2", "url": "http://example.com/2"}
    ]
    clusters, num_clusters, embeddings = cluster_texts(bookmarks)
    
    assert isinstance(clusters, np.ndarray), "Clusters should be a numpy array"
    assert isinstance(num_clusters, int), "Number of clusters should be an integer"
    assert embeddings.shape == (2, 768), "Embeddings shape should match the number of bookmarks and embedding size"