from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

app = Flask(__name__)

# Load the SBERT model globally to avoid reloading it on each request
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/cluster', methods=['POST'])
def cluster_texts():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the CSV file. Assuming no header, as we are working with a single row of texts.
        df = pd.read_csv(file, header=None)
        # Convert the first row to a list, handling the case where there might be multiple columns of text.
        texts = df.iloc[0, :].tolist()

        print(f"Clustering {len(texts)} texts")

        # Generate embeddings
        embeddings = model.encode(texts)

        # Determine the number of clusters, with a maximum of 8
        num_clusters = min(8, len(set(texts)))
        print(f"Clustering {len(texts)} texts into {num_clusters} clusters")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Prepare cluster information
        cluster_info = [{"text": text, "cluster": int(cluster)} for text, cluster in zip(texts, clusters)]

        return jsonify(cluster_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
