from flask import request, jsonify, current_app as app
from flask_cors import cross_origin
import json

from .cluster_bookmarks import cluster_texts
from .cluster_naming import generate_cluster_names
from .bookmark_import import build_json_import

@app.route('/cluster', methods=['POST'])
@cross_origin()
def cluster_texts_from_json():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    json_data = request.get_json()

    try:
        # Correctly log the type and optionally the length or other properties of json_data
        app.logger.debug(f"Received JSON data of type {type(json_data).__name__}, length: {len(json_data) if isinstance(json_data, list) else 'N/A'}")
        
        clusters, num_clusters, embeddings = cluster_texts(json_data)
        cluster_info = generate_cluster_names(json_data, clusters, num_clusters)
        bookmark_import = build_json_import(cluster_info)
        return jsonify(bookmark_import)
    except json.JSONDecodeError as json_err:
        return jsonify({"error": f"Malformed JSON data: {json_err}"}), 400
    except ValueError as val_err:
        return jsonify({"error": f"Value error: {val_err}"}), 400
    except KeyError as key_err:
        app.logger.error(f"Missing key in JSON data: {key_err}")
        return jsonify({"error": f"Missing key: {key_err}"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
