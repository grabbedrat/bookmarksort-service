from flask import request, jsonify, current_app as app
# Assuming you have similar functions to process JSON data
from .cluster_bookmarks import cluster_texts
from .cluster_naming import generate_cluster_names
from .bookmark_import import build_json_import
import json

@app.route('/cluster', methods=['POST', 'OPTIONS'])
def cluster_texts_from_json():
    # Check if there is JSON data in the request
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    # Get JSON data
    json_data = request.get_json()
    
    try:
        # Process the JSON data, e.g., to extract text or relevant information
        #print("Received data: ", json_data)
        print("process_json_data: ", json_data)
        
        # Cluster the extracted texts
        print("process_json_data done, clustering...")
        clusters, num_clusters, embeddings = cluster_texts(json_data)
        
        # Generate names or other relevant information for the clusters
        print("clustering done, generating names...")
        cluster_info = generate_cluster_names(json_data, clusters, num_clusters)

        print("names generated, formatting...")
        bookmark_import = build_json_import(cluster_info)
        
        return jsonify(bookmark_import)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
