from flask import request, jsonify, current_app as app
from flask_cors import cross_origin
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils.cluster_naming import generate_cluster_names
from .utils.bookmark_import import build_json_import
from .utils.vis_clusters import visualize_clusters

@app.route('/cluster', methods=['POST'])
@cross_origin()
def cluster_texts_from_json():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    json_data = request.get_json()
    if not isinstance(json_data, list):
        return jsonify({"error": "Expected a list of bookmark objects"}), 400

    expected_keys = {"name", "url"}
    for bookmark in json_data:
        if not expected_keys.issubset(bookmark):
            return jsonify({"error": "Bookmark object missing required keys"}), 400

    try:
        # Extract text data from bookmarks
        texts = [bookmark["name"] for bookmark in json_data]

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Perform clustering using HDBSCAN with a lower min_cluster_size
        clusterer = HDBSCAN(min_cluster_size=3)  # Adjust this value as needed
        cluster_labels = clusterer.fit_predict(X)

        # Assign bookmarks to clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Exclude noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(json_data[i])

        num_clusters = len(clusters)
        app.logger.info("Clusters clustered")

        cluster_info = generate_cluster_names(clusters)
        structured_data = build_structured_json(clusters, cluster_info)

        output_file = "cluster_visualization.png"
        visualize_clusters(clusters, num_clusters, X.toarray(), output_file)

        return jsonify(structured_data)

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

def build_structured_json(clusters, cluster_info):
    structured_data = []

    def create_folder(cluster_id, parent_id=None):
        # Convert NumPy int64 to Python int
        cluster_id = int(cluster_id)

        folder = {
            "type": "folder",
            "id": cluster_id,
            "name": cluster_info[cluster_id],
            "children": []
        }

        if parent_id is not None:
            folder["parent_id"] = int(parent_id)

        for bookmark in clusters[cluster_id]:
            bookmark_name = bookmark["name"].strip()
            if not bookmark_name:
                bookmark_name = "Untitled Bookmark"

            folder["children"].append({
                "type": "bookmark",
                "name": bookmark_name,
                "url": bookmark["url"]
            })

        return folder

    def assign_parent_folders(folder, parent_id, depth=0, max_depth=5):
        if depth > max_depth:
            structured_data.append(folder)  # Add folder to top-level if max depth is exceeded
            return

        if parent_id is not None:
            parent_folder = next((f for f in structured_data if f["type"] == "folder" and f.get("id") == parent_id), None)
            if parent_folder:
                parent_folder["children"].append(folder)
            else:
                assign_parent_folders(folder, parent_id // 2, depth + 1)  # Recursively search for parent folder
        else:
            structured_data.append(folder)

    for cluster_id in clusters:
        folder = create_folder(cluster_id)
        parent_id = cluster_id // 2  # Assign parent folder based on cluster ID
        assign_parent_folders(folder, parent_id)

    return structured_data

if __name__ == "__main__":
    app.run(debug=True)