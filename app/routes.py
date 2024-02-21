from flask import request, jsonify, current_app as app
from .text_processing import process_text_file
from .clustering import cluster_texts
from .cluster_naming import generate_cluster_names

@app.route('/cluster', methods=['POST'])
def cluster_texts_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        texts = process_text_file(file)
        clusters, num_clusters = cluster_texts(texts)
        cluster_info = generate_cluster_names(texts, clusters, num_clusters)
        return jsonify(cluster_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
