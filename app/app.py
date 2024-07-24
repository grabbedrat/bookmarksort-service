from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields, reqparse
from flask_cors import CORS
from bertopic import BERTopic
import umap
import hdbscan
import numpy as np
from typing import List, Dict
import logging
import threading
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, PickleType, func
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
        self.embedding_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self.topic_model = None
        self.is_ready = False
        self.is_initializing = False

    def initialize(self, embedding_model="all-MiniLM-L6-v2", umap_n_neighbors=15, umap_n_components=5, 
                   umap_min_dist=0.0, hdbscan_min_cluster_size=15, hdbscan_min_samples=10, 
                   nr_topics="auto", top_n_words=10):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info("Initializing models...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.umap_model = umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, 
                                        min_dist=umap_min_dist, metric='cosine')
            self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, 
                                                 min_samples=hdbscan_min_samples, metric='euclidean', 
                                                 cluster_selection_method='eom')
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                nr_topics=nr_topics,
                top_n_words=top_n_words
            )
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
        
        # Embedding stage
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Clustering and topic modeling stage
        topics, _ = self.topic_model.fit_transform(texts, embeddings)

        organized_bookmarks = {}
        session = Session()
        try:
            for bookmark, topic, embedding in zip(bookmarks, topics, embeddings):
                topic_name = f"Topic_{topic}" if topic != -1 else "Uncategorized"
                if topic_name not in organized_bookmarks:
                    organized_bookmarks[topic_name] = []
                
                bookmark_data = {
                    "url": bookmark["url"],
                    "title": bookmark["title"],
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
                    logger.warning(f"Bookmark with URL {bookmark['url']} already exists. Skipping.")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing bookmarks: {str(e)}")
            raise
        finally:
            session.close()

        return organized_bookmarks

    def add_bookmark(self, bookmark: Dict) -> Dict:
        return self.process_bookmarks([bookmark])

    def list_bookmarks(self, topic: str = None, page: int = 1, per_page: int = 20) -> Dict:
        session = Session()
        try:
            query = session.query(Bookmark)
            if topic:
                query = query.filter(Bookmark.topic == topic)
            
            total = query.count()
            bookmarks = query.offset((page - 1) * per_page).limit(per_page).all()
            
            return {
                "bookmarks": [
                    {
                        "url": bookmark.url,
                        "title": bookmark.title,
                        "topic": bookmark.topic
                    }
                    for bookmark in bookmarks
                ],
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        finally:
            session.close()

    def search_bookmarks(self, query: str) -> List[Dict]:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

        session = Session()
        try:
            all_bookmarks = session.query(Bookmark).all()
            results = []
            for bookmark in all_bookmarks:
                similarity = np.dot(query_embedding, np.array(bookmark.embedding))
                results.append((bookmark, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return [
                {
                    "url": bookmark.url,
                    "title": bookmark.title,
                    "topic": bookmark.topic,
                    "similarity": float(similarity)
                }
                for bookmark, similarity in results[:10]  # Return top 10 results
            ]
        finally:
            session.close()

    def get_topics(self) -> List[Dict]:
        session = Session()
        try:
            topics = session.query(Bookmark.topic, func.count(Bookmark.id)).group_by(Bookmark.topic).all()
            return [{"topic": topic, "count": count} for topic, count in topics]
        finally:
            session.close()

    def get_visualization_data(self) -> Dict:
        session = Session()
        try:
            bookmarks = session.query(Bookmark).all()
            
            # Create a 2D projection of the embeddings
            embeddings = np.array([bookmark.embedding for bookmark in bookmarks])
            pca = PCA(n_components=2)
            projections = pca.fit_transform(embeddings)
            
            visualization_data = {
                "nodes": [
                    {
                        "id": bookmark.id,
                        "url": bookmark.url,
                        "title": bookmark.title,
                        "topic": bookmark.topic,
                        "x": float(projections[i][0]),
                        "y": float(projections[i][1])
                    }
                    for i, bookmark in enumerate(bookmarks)
                ],
                "links": []  # You can add links between related bookmarks if needed
            }
            return visualization_data
        finally:
            session.close()

organizer = BookmarkOrganizer()

# API models
bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

bookmark_response_model = api.model('BookmarkResponse', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic')
})

bookmarks_list_model = api.model('BookmarksList', {
    'bookmarks': fields.List(fields.Nested(bookmark_model)),
    'total': fields.Integer(description='Total number of bookmarks'),
    'page': fields.Integer(description='Current page number'),
    'per_page': fields.Integer(description='Number of bookmarks per page'),
    'total_pages': fields.Integer(description='Total number of pages')
})

search_result_model = api.model('SearchResult', {
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'topic': fields.String(description='The bookmark topic'),
    'similarity': fields.Float(description='Similarity score')
})

topic_model = api.model('Topic', {
    'topic': fields.String(description='Topic name'),
    'count': fields.Integer(description='Number of bookmarks in this topic')
})

@ns.route('/process')
class ProcessBookmarks(Resource):
    @ns.expect([bookmark_model])
    @ns.marshal_with(bookmark_response_model)
    def post(self):
        """Process a list of bookmarks and organize them by topics"""
        if not organizer.is_ready:
            return {"success": False, "error": "Model is still initializing. Please try again later."}, 503

        bookmarks = api.payload
        try:
            result = organizer.process_bookmarks(bookmarks)
            return {"success": True, "organized_bookmarks": result}
        except Exception as e:
            logger.error(f"Error processing bookmarks: {str(e)}")
            return {"success": False, "error": str(e)}, 500

@ns.route('/add')
class AddBookmark(Resource):
    @ns.expect(bookmark_model)
    @ns.marshal_with(bookmark_response_model)
    def post(self):
        """Add a new bookmark"""
        if not organizer.is_ready:
            return {"success": False, "error": "Model is still initializing. Please try again later."}, 503

        bookmark = api.payload
        result = organizer.add_bookmark(bookmark)
        return {"success": True, "organized_bookmarks": result}

@ns.route('/list')
@ns.param('topic', 'Filter bookmarks by topic (optional)')
@ns.param('page', 'Page number (default: 1)')
@ns.param('per_page', 'Number of bookmarks per page (default: 20)')
class ListBookmarks(Resource):
    @ns.marshal_with(bookmarks_list_model)
    def get(self):
        """List all bookmarks, optionally filtered by topic"""
        topic = request.args.get('topic')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        return organizer.list_bookmarks(topic, page, per_page)

@ns.route('/search')
@ns.param('q', 'Search query')
class SearchBookmarks(Resource):
    @ns.marshal_with(search_result_model)
    def get(self):
        """Search bookmarks by keyword"""
        if not organizer.is_ready:
            return {"error": "Model is still initializing. Please try again later."}, 503

        query = request.args.get('q')
        if not query:
            return {"error": "Search query is required"}, 400

        return organizer.search_bookmarks(query)

@ns.route('/topics')
class Topics(Resource):
    @ns.marshal_with(topic_model)
    def get(self):
        """Get all topics and their bookmark counts"""
        return organizer.get_topics()

@ns.route('/visualization')
class Visualization(Resource):
    def get(self):
        """Get visualization data for bookmarks"""
        if not organizer.is_ready:
            return {"error": "Model is still initializing. Please try again later."}, 503
        
        try:
            visualization_data = organizer.get_visualization_data()
            return {"success": True, "visualization_data": visualization_data}
        except Exception as e:
            logger.error(f"Error getting visualization data: {str(e)}")
            return {"success": False, "error": str(e)}, 500

@app.route('/status')
def status():
    if organizer.is_ready:
        return {"status": "ready"}, 200
    elif organizer.is_initializing:
        return {"status": "initializing"}, 202
    else:
        return {"status": "not started"}, 500

def initialize_model_async():
    organizer.initialize(
        embedding_model="all-MiniLM-L6-v2",
        umap_n_neighbors=15,
        umap_n_components=5,
        umap_min_dist=0.0,
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=10,
        nr_topics="auto",
        top_n_words=10
    )

# Start model initialization in a separate thread
threading.Thread(target=initialize_model_async).start()

if __name__ == '__main__':
    app.run(debug=True)