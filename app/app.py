from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from bertopic import BERTopic
import umap
import numpy as np
from typing import List, Dict
import logging
import threading
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Bookmark Organizer API',
          description='An API for organizing bookmarks using BERTopic')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ns = api.namespace('bookmarks', description='Bookmark operations')

class BookmarkOrganizer:
    def __init__(self):
        self.model = None
        self.umap_model = None
        self.sentence_transformer = None
        self.is_ready = False
        self.is_initializing = False

    def initialize(self):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info("Initializing BERTopic model...")
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.model = BERTopic(language="english", min_topic_size=5, nr_topics="auto", embedding_model=self.sentence_transformer)
            self.umap_model = umap.UMAP(n_components=2, random_state=42)
            self.is_ready = True
            logger.info("Model initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
        finally:
            self.is_initializing = False

    def process_bookmarks(self, bookmarks: List[Dict]) -> Dict:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        texts = [f"{b['title']} {b['url']}" for b in bookmarks]
        topics, probs = self.model.fit_transform(texts)
        embeddings = self.sentence_transformer.encode(texts)
        reduced_embeddings = self.umap_model.fit_transform(embeddings)

        organized_bookmarks = {}
        for bookmark, topic, coord in zip(bookmarks, topics, reduced_embeddings):
            topic_name = f"Topic_{topic}" if topic != -1 else "Uncategorized"
            if topic_name not in organized_bookmarks:
                organized_bookmarks[topic_name] = []
            organized_bookmarks[topic_name].append({
                "url": bookmark["url"],
                "title": bookmark["title"],
                "x": float(coord[0]),
                "y": float(coord[1])
            })

        return {
            "organized_bookmarks": organized_bookmarks,
            "reduced_embeddings": reduced_embeddings.tolist()
        }

organizer = BookmarkOrganizer()

bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

organized_bookmarks_model = api.model('OrganizedBookmarks', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic'),
    'reduced_embeddings': fields.List(fields.List(fields.Float), description='UMAP reduced embeddings')
})

@ns.route('/process')
class BookmarkProcess(Resource):
    @ns.expect([bookmark_model])
    @ns.marshal_with(organized_bookmarks_model)
    def post(self):
        """Process a list of bookmarks and organize them by topics"""
        if not organizer.is_ready:
            api.abort(503, "Model is still initializing. Please try again later.")

        bookmarks = api.payload
        try:
            result = organizer.process_bookmarks(bookmarks)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error processing bookmarks: {str(e)}")
            api.abort(500, f"Error processing bookmarks: {str(e)}")

@app.route('/status')
def status():
    if organizer.is_ready:
        return {"status": "ready"}, 200
    elif organizer.is_initializing:
        return {"status": "initializing"}, 202
    else:
        return {"status": "not started"}, 500

def initialize_model_async():
    organizer.initialize()

# Start model initialization in a separate thread
threading.Thread(target=initialize_model_async).start()

if __name__ == '__main__':
    app.run(debug=True)