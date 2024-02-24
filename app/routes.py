from flask import request, jsonify, current_app as app
# Assuming you have similar functions to process JSON data
from .json_processing import process_json_data
from .cluster_bookmarks import cluster_texts
from .cluster_naming import generate_cluster_names
from .bookmark_import import build_html_import

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
        bookmarkdata = process_json_data(json_data)
        
        # Cluster the extracted texts
        print("process_json_data done, clustering...")
        clusters, num_clusters, embeddings = cluster_texts(bookmarkdata)
        
        # Generate names or other relevant information for the clusters
        print("clustering done, generating names...")
        cluster_info = generate_cluster_names(bookmarkdata, clusters, num_clusters)

       # Format to importable html format
        print("names generated, formatting...")
        bookmark_import = build_html_import(cluster_info, "./clustered_bookmarks.html")

        print("saved")
        
        return jsonify(bookmark_import)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
