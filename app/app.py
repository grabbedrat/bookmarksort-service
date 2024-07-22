from flask import Flask
from flask_restx import Api, Resource, fields
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import logging
import threading
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
api = Api(app, version='1.0', title='Bookmark Sorting API',
          description='An API for sorting and categorizing bookmarks using BERTopic')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ns = api.namespace('bookmarks', description='Bookmark operations')

@dataclass
class Bookmark:
    url: str
    title: str
    tags: List[str] = field(default_factory=list)

    def to_document(self):
        return f"{self.title} {self.url} {' '.join(self.tags)}".strip()

@dataclass
class BookmarkResult(Bookmark):
    topics: List[str] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)

class BERTopicWrapper:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.documents = []
        self.is_ready = False
        self.is_initializing = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info(f"Initializing BERTopic model on {self.device}...")
        try:
            # Explicitly set the device for SentenceTransformer
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
            self.model = BERTopic(
                language="english",
                min_topic_size=5,
                nr_topics="auto",
                calculate_probabilities=True,
                verbose=True,
                embedding_model=embedding_model
            )
            self.vectorizer = CountVectorizer(stop_words="english")
            self.is_ready = True
            logger.info(f"BERTopic model initialized successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to initialize BERTopic model: {str(e)}")
        finally:
            self.is_initializing = False

    def process_bookmarks(self, bookmarks):
        if not self.is_ready:
            raise RuntimeError("BERTopic model is not yet initialized")

        new_documents = [bookmark.to_document() for bookmark in bookmarks]
        self.documents.extend(new_documents)
        
        topics, probs = self.model.fit_transform(new_documents)
        
        results = []
        for bookmark, topic, prob in zip(bookmarks, topics, probs):
            topic_labels = self.model.get_topic(topic)
            topic_names = [label[0] for label in topic_labels if label]
            results.append(BookmarkResult(
                url=bookmark.url,
                title=bookmark.title,
                tags=bookmark.tags,
                topics=topic_names,
                probabilities=prob.tolist()
            ))
        
        return results

    def organize_bookmarks(self, bookmarks):
        results = self.process_bookmarks(bookmarks)
        organized_bookmarks = {}
        
        for result in results:
            main_topic = result.topics[0] if result.topics else "Uncategorized"
            if main_topic not in organized_bookmarks:
                organized_bookmarks[main_topic] = []
            organized_bookmarks[main_topic].append({
                "url": result.url,
                "title": result.title,
                "tags": result.tags
            })
        
        return organized_bookmarks

# Global variable to hold the BERTopicWrapper instance
topic_wrapper = BERTopicWrapper()

def initialize_model_async():
    topic_wrapper.initialize()

# Start model initialization in a separate thread
threading.Thread(target=initialize_model_async).start()

bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title'),
    'tags': fields.List(fields.String, description='List of tags associated with the bookmark')
})

organized_bookmarks_model = api.model('OrganizedBookmarks', {
    'topic': fields.List(fields.Nested(bookmark_model), description='List of bookmarks for each topic')
})

@ns.route('/')
class BookmarksResource(Resource):
    @ns.expect([bookmark_model])
    def post(self):
        """Process a list of bookmarks and organize them by topics"""
        if not topic_wrapper.is_ready:
            return {"success": False, "error": "Model is still initializing. Please try again later."}, 503

        bookmarks_data = api.payload
        if not bookmarks_data:
            return {"success": False, "error": "No bookmarks provided"}, 400

        bookmarks = [Bookmark(url=b['url'], title=b['title'], tags=b.get('tags', [])) for b in bookmarks_data]
        try:
            organized_bookmarks = topic_wrapper.organize_bookmarks(bookmarks)
            
            formatted_bookmarks = {}
            for topic, bookmarks_list in organized_bookmarks.items():
                formatted_bookmarks[topic] = [
                    {"url": b["url"], "title": b["title"], "tags": b.get("tags", [])}
                    for b in bookmarks_list
                ]
            
            return {"success": True, "preview": formatted_bookmarks}
        except Exception as e:
            logger.error(f"Error processing bookmarks: {str(e)}")
            return {"success": False, "error": str(e)}, 500

@app.route('/status')
def model_status():
    if topic_wrapper.is_ready:
        return {"status": "ready", "device": str(topic_wrapper.device)}, 200
    elif topic_wrapper.is_initializing:
        return {"status": "initializing", "device": str(topic_wrapper.device)}, 202
    else:
        return {"status": "not started"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)