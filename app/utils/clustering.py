import time
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import logging
from .config import client

# Disable parallel tokenization to avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_name):
    """
    Load a pre-trained SentenceTransformer model.

    Args:
        model_name (str): Name of the pre-trained model.

    Returns:
        SentenceTransformer: Loaded SentenceTransformer model.

    Raises:
        RuntimeError: If an error occurs during model loading.
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def validate_bookmarks(bookmarks):
    """
    Validate the format and content of the provided bookmarks.

    Args:
        bookmarks (list): List of bookmark dictionaries.

    Raises:
        ValueError: If no bookmarks are provided or the bookmarks format is incorrect.
    """
    if not bookmarks:
        raise ValueError("No bookmarks provided.")
    if not isinstance(bookmarks, list) or not all(isinstance(item, dict) for item in bookmarks):
        raise ValueError(f"Incorrect bookmarks format: {bookmarks}")

def encode_texts(model, texts):
    """
    Encode a list of texts using the provided SentenceTransformer model.

    Args:
        model (SentenceTransformer): Pre-loaded SentenceTransformer model.
        texts (list): List of texts to encode.

    Returns:
        list: Encoded text embeddings.

    Raises:
        RuntimeError: If an error occurs during text encoding.
    """
    try:
        return model.encode(texts)
    except Exception as e:
        raise RuntimeError(f"Error encoding texts: {e}")

def perform_clustering(embeddings, distance_threshold=2):
    """
    Perform hierarchical agglomerative clustering on the provided embeddings.

    Args:
        embeddings (list): List of text embeddings.
        distance_threshold (float, optional): Distance threshold for clustering. Defaults to 2.

    Returns:
        tuple: A tuple containing the cluster assignments and linkage matrix.

    Raises:
        RuntimeError: If an error occurs during clustering.
    """
    try:
        link_matrix = linkage(embeddings, method='ward')
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, compute_full_tree=True)
        clusters = clustering.fit_predict(embeddings)
        return clusters, link_matrix
    except Exception as e:
        raise RuntimeError(f"Error during clustering: {e}")

def cluster_texts(bookmarks):
    """
    Cluster the provided bookmarks based on their text content.

    Args:
        bookmarks (list): List of bookmark dictionaries.

    Returns:
        tuple: A tuple containing the cluster assignments, number of clusters, text embeddings,
               and linkage matrix.
    """
    validate_bookmarks(bookmarks)
    model = load_model('all-MiniLM-L6-v2')
    texts = [f"{bookmark['name']} {bookmark['url']}" for bookmark in bookmarks]
    embeddings = encode_texts(model, texts)
    clusters, link_matrix = perform_clustering(embeddings)
    return clusters, len(set(clusters)), embeddings, link_matrix

def generate_cluster_names(clusters):
    """
    Generate descriptive names for the clusters using OpenAI's GPT-3.5-turbo model.

    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of bookmarks.

    Returns:
        dict: Dictionary mapping cluster IDs to generated cluster names.
    """
    cluster_info = {}
    for cluster_id in clusters:
        prompt = f"Generate a concise and descriptive name for a cluster of bookmarks with the following tags:\n\n"
        for bookmark in clusters[cluster_id]:
            tags = ", ".join(bookmark.get("tags", []))
            prompt += f"- {tags}\n"
        prompt += "\nCluster Name:"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates concise and descriptive names for clusters of bookmarks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )
            if response.choices:
                cluster_name = response.choices[0].message.content.strip()
                cluster_name = postprocess_name(cluster_name)  # Postprocess the generated name
                cluster_info[cluster_id] = cluster_name
            else:
                cluster_info[cluster_id] = f"Cluster {cluster_id}"
        except Exception as e:
            logging.error(f"Error generating name for cluster {cluster_id}: {e}")
            cluster_info[cluster_id] = f"Cluster {cluster_id}"

    return cluster_info

def preprocess_description(name, url):
    """
    Preprocess the bookmark description by combining the name and URL,
    removing special characters, and truncating if necessary.

    Args:
        name (str): Bookmark name.
        url (str): Bookmark URL.

    Returns:
        str: Preprocessed bookmark description.
    """
    # Combine the name and URL into a single description
    description = f"{name} - {url}"
    
    # Remove URLs and special characters from the description
    cleaned_description = re.sub(r'http\S+', '', description)
    cleaned_description = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_description)
    
    # Truncate long descriptions
    max_length = 100
    if len(cleaned_description) > max_length:
        cleaned_description = cleaned_description[:max_length] + "..."
    
    return cleaned_description

def postprocess_name(name):
    """
    Postprocess the generated cluster name by removing irrelevant or incoherent parts.

    Args:
        name (str): Generated cluster name.

    Returns:
        str: Postprocessed cluster name.
    """
    # Remove any irrelevant or incoherent parts from the generated name
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    name = name.strip()
    
    return name