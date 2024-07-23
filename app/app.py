from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from bertopic import BERTopic
import umap
import numpy as np
from typing import List, Dict
import logging
import threading
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app, version='1.0', title='Bookmark Organizer API',
          description='An API for organizing bookmarks using BERTopic')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ns = api.namespace('bookmarks', description='Bookmark operations')

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///bookmarks.db', echo=True)
Session = sessionmaker(bind=engine)

class Bookmark(Base):
    __tablename__ = 'bookmarks'

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    embedding = Column(PickleType, nullable=False)
    topic = Column(String, nullable=False)

Base.metadata.create_all(engine)

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
        topics, _ = self.model.fit_transform(texts)
        embeddings = self.sentence_transformer.encode(texts)
        
        try:
            reduced_embeddings = self.umap_model.fit_transform(embeddings)
        except ValueError:
            # If UMAP fails, use PCA as a fallback
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

        organized_bookmarks = {}
        session = Session()
        try:
            for bookmark, topic, embedding, reduced_emb in zip(bookmarks, topics, embeddings, reduced_embeddings):
                topic_name = f"Topic_{topic}" if topic != -1 else "Uncategorized"
                if topic_name not in organized_bookmarks:
                    organized_bookmarks[topic_name] = []
                
                bookmark_data = {
                    "url": bookmark["url"],
                    "title": bookmark["title"],
                    "embedding": reduced_emb.tolist()
                }
                organized_bookmarks[topic_name].append(bookmark_data)

                # Store in database
                try:
                    db_bookmark = Bookmark(url=bookmark["url"], title=bookmark["title"], 
                                           embedding=embedding.tolist(), topic=topic_name)
                    session.merge(db_bookmark)  # merge will insert or update
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    # If the bookmark already exists, we'll just skip it
                    logger.warning(f"Bookmark with URL {bookmark['url']} already exists. Skipping.")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing bookmarks: {str(e)}")
            raise
        finally:
            session.close()

        return {
            "organized_bookmarks": organized_bookmarks
        }

    def find_similar_bookmarks(self, bookmark_url: str, top_k: int = 5) -> List[Dict]:
        session = Session()
        try:
            # Fetch the query bookmark
            query_bookmark = session.query(Bookmark).filter_by(url=bookmark_url).first()
            if not query_bookmark:
                return []

            query_embedding = np.array(query_bookmark.embedding)

            # Fetch all bookmarks
            all_bookmarks = session.query(Bookmark).all()

            # Calculate similarities
            similarities = []
            for bookmark in all_bookmarks:
                if bookmark.url != bookmark_url:
                    similarity = np.dot(query_embedding, np.array(bookmark.embedding))
                    similarities.append((bookmark, similarity))

            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:top_k]

            return [
                {
                    "url": bookmark.url,
                    "title": bookmark.title,
                    "topic": bookmark.topic,
                    "similarity": float(similarity)
                }
                for bookmark, similarity in top_similar
            ]
        finally:
            session.close()

organizer = BookmarkOrganizer()

# API models
bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

organized_bookmarks_model = api.model('OrganizedBookmarks', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic with embeddings')
})

@ns.route('/process')
class BookmarkProcess(Resource):
    @ns.expect([bookmark_model])
    @ns.marshal_with(organized_bookmarks_model)
    def post(self):
        """Process a list of bookmarks and organize them by topics"""
        if not organizer.is_ready:
            return jsonify({"success": False, "error": "Model is still initializing. Please try again later."}), 503

        bookmarks = api.payload
        try:
            result = organizer.process_bookmarks(bookmarks)
            return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error processing bookmarks: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

@ns.route('/similar/<string:bookmark_url>')
class SimilarBookmarks(Resource):
    def get(self, bookmark_url):
        """Find similar bookmarks based on the given bookmark URL"""
        similar_bookmarks = organizer.find_similar_bookmarks(bookmark_url)
        
        if not similar_bookmarks:
            return {"success": False, "message": "Bookmark not found"}, 404
        
        return {"success": True, "similar_bookmarks": similar_bookmarks}

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