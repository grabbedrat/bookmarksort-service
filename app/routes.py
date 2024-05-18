from flask import request, jsonify, current_app as app
from flask_cors import cross_origin
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils.clustering import generate_cluster_names
from .utils.visualization import visualize_clusters, visualize_structured_data
from .utils.data_preprocessing import clean_bookmark_data, add_tags_to_bookmarks
from .utils.json_builder import build_structured_json

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
        # Clean the bookmark data
        cleaned_data = clean_bookmark_data(json_data)

        # Add tags to the cleaned bookmark data
        tagged_data = add_tags_to_bookmarks(cleaned_data)
        print(f"Number of bookmarks received: {len(tagged_data)}")

        # Extract text data from bookmarks
        texts = [bookmark["name"] for bookmark in tagged_data]

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Perform clustering using HDBSCAN
        clusterer = HDBSCAN()
        cluster_labels = clusterer.fit_predict(X)

        # Assign bookmarks to clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Exclude noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(tagged_data[i])

        num_clusters = len(clusters)
        print(f"Bookmarks sorted into {num_clusters} clusters")

        cluster_info = generate_cluster_names(clusters)
        structured_data = build_structured_json(clusters, cluster_info)

        output_file = "cluster_visualization.png"
        visualize_clusters(clusters, num_clusters, X.toarray(), output_file)

        # Visualize the structured data and save as JPEG
        output_file_structured = "structured_data_visualization.jpeg"
        visualize_structured_data(structured_data, output_file_structured)

        print(f"{len(json_data)} bookmarks sorted into {len(clusters)} clusters")

        return jsonify(structured_data)

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)