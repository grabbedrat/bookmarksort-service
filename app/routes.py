from flask import request, jsonify, current_app as app
from flask_cors import cross_origin

from .clustering_methods.heirarchical_clustering import cluster_texts as cluster_texts_hier
from .utils.cluster_naming import generate_cluster_names
from .utils.bookmark_import import build_json_import
from .utils.vis_clusters import visualize_clusters, plot_dendrogram

@app.route('/cluster', methods=['POST'])
@cross_origin()
def cluster_texts_from_json():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    json_data = request.get_json()

    if not isinstance(json_data, list):
        return jsonify({"error": "Expected a list of bookmark objects"}), 400

    expected_keys = {"name", "url", "tags"}
    for bookmark in json_data:
        if not expected_keys.issubset(bookmark):
            return jsonify({"error": "Bookmark object missing required keys"}), 400
        if not isinstance(bookmark['tags'], list):
            return jsonify({"error": "Tags must be a list"}), 400

    try:
        clusters, num_clusters, embeddings, link_matrix = cluster_texts_hier(json_data)
        if link_matrix is not None:
            output_dendrogram_file = "dendrogram_visualization.png"
            plot_dendrogram(link_matrix, output_dendrogram_file)
            app.logger.info(f"Dendrogram saved to {output_dendrogram_file}")

        app.logger.info("Clusters clustered")
        cluster_info = generate_cluster_names(json_data, clusters, num_clusters)
        bookmark_import = build_json_import(cluster_info)

        output_file = "cluster_visualization.png"
        visualize_clusters(clusters, num_clusters, embeddings, output_file)

        return jsonify(bookmark_import)
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)
