from flask import request, jsonify, current_app as app
# Assuming you have similar functions to process JSON data
from .json_processing import process_json_data
from .clustering import cluster_texts
from .cluster_naming import generate_cluster_names

@app.route('/cluster', methods=['POST'])
def cluster_texts_from_json():
    # Check if there is JSON data in the request
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    # Get JSON data
    json_data = request.get_json()
    
    try:
        # Process the JSON data, e.g., to extract text or relevant information
        print("Received data: ", json_data)
        texts = process_json_data(json_data)
        
        # Cluster the extracted texts
        print("process_json_data done, clustering...")
        clusters, num_clusters = cluster_texts(texts)
        
        # Generate names or other relevant information for the clusters
        print("clustering done, generating names...")
        cluster_info = generate_cluster_names(texts, clusters, num_clusters)

        print("names generated, returning...")
        
        return jsonify(cluster_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
