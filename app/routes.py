from flask import request, jsonify, current_app as app
from flask_cors import cross_origin
import json

from .clustering_methods.kmeans_clustering import cluster_texts
from .utils.cluster_naming import generate_cluster_names
from .utils.bookmark_import import build_json_import

@app.route('/cluster', methods=['POST'])
@cross_origin()
def cluster_texts_from_json():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    json_data = request.get_json()
    # app.logger.debug(f"Incoming request data: {request.data}")

    # Shape/Format Checks
    if not isinstance(json_data, list):
        return jsonify({"error": "Expected a list of bookmark objects"}), 400

    # Optional: Check for the existence of expected keys in each bookmark object
    expected_keys = {"name", "url", "tags"}
    for bookmark in json_data:
        if not all(key in bookmark for key in expected_keys):
            return jsonify({"error": "Bookmark object missing required keys"}), 400
        if not isinstance(bookmark['tags'], list):
            return jsonify({"error": "Tags must be a list"}), 400

    try:
        clusters, num_clusters, embeddings = cluster_texts(json_data)
        cluster_info = generate_cluster_names(json_data, clusters, num_clusters)
        bookmark_import = build_json_import(cluster_info)

        # It's better to directly log the generated data rather than attempting to read the data attribute from jsonify
        # app.logger.debug(f"Outgoing response data: {bookmark_import}")

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

if __name__ == "__main__":
    app.run(debug=True)