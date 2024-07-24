# bookmark_organizer.py
import logging
import threading
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan
from sklearn.decomposition import PCA
from models import db, Bookmark
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

logger = logging.getLogger(__name__)

class BookmarkOrganizer:
    def __init__(self):
        self.embedding_model = None
        self.umap_model = None
        self.hdbscan_model = None
        self.topic_model = None
        self.is_ready = False
        self.is_initializing = False
        self.params = {
            "embedding_model": "all-MiniLM-L6-v2",
            "umap_n_neighbors": 15,
            "umap_n_components": 5,
            "umap_min_dist": 0.0,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 10,
            "nr_topics": "auto",
            "top_n_words": 10
        }

    def initialize(self, **kwargs):
        if self.is_initializing:
            return
        self.is_initializing = True
        logger.info("Initializing models...")
        try:
            # Update params with any provided kwargs
            self.params.update(kwargs)

            self.embedding_model = SentenceTransformer(self.params["embedding_model"])
            self.umap_model = umap.UMAP(
                n_neighbors=self.params["umap_n_neighbors"],
                n_components=self.params["umap_n_components"],
                min_dist=self.params["umap_min_dist"],
                metric='cosine'
            )
            self.hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=self.params["hdbscan_min_cluster_size"],
                min_samples=self.params["hdbscan_min_samples"],
                metric='euclidean',
                cluster_selection_method='eom'
            )
            self.topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                nr_topics=self.params["nr_topics"],
                top_n_words=self.params["top_n_words"]
            )
            self.is_ready = True
            logger.info("Model initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
        finally:
            self.is_initializing = False

    def update_parameters(self, new_params):
        self.params.update(new_params)
        self.is_ready = False
        self.initialize()

    def process_bookmarks(self) -> Dict:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        bookmarks = Bookmark.query.all()
        texts = [f"{b.title} {b.url}" for b in bookmarks]
        
        # Embedding stage
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Clustering and topic modeling stage
        topics, _ = self.topic_model.fit_transform(texts, embeddings)

        organized_bookmarks = {}
        try:
            for bookmark, topic, embedding in zip(bookmarks, topics, embeddings):
                topic_name = f"Topic_{topic}" if topic != -1 else "Uncategorized"
                if topic_name not in organized_bookmarks:
                    organized_bookmarks[topic_name] = []
                
                bookmark_data = {
                    "url": bookmark.url,
                    "title": bookmark.title,
                }
                organized_bookmarks[topic_name].append(bookmark_data)

                # Update bookmark in database
                bookmark.embedding = embedding.tolist()
                bookmark.topic = topic_name

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing bookmarks: {str(e)}")
            raise

        return organized_bookmarks

    def add_bookmark(self, bookmark: Dict) -> Dict:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        text = f"{bookmark['title']} {bookmark['url']}"
        
        # Embedding stage
        embedding = self.embedding_model.encode([text], show_progress_bar=False)[0]
        
        # Predict topic
        topic, _ = self.topic_model.transform([text])
        topic_name = f"Topic_{topic[0]}" if topic[0] != -1 else "Uncategorized"

        try:
            db_bookmark = Bookmark(
                url=bookmark["url"],
                title=bookmark["title"],
                embedding=embedding.tolist(),
                topic=topic_name
            )
            db.session.add(db_bookmark)
            db.session.commit()

            return {topic_name: [{"url": bookmark["url"], "title": bookmark["title"]}]}
        except IntegrityError:
            db.session.rollback()
            logger.warning(f"Bookmark with URL {bookmark['url']} already exists. Skipping.")
            return {}
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding bookmark: {str(e)}")
            raise

    def list_bookmarks(self, topic: str = None, page: int = 1, per_page: int = 20) -> Dict:
        query = Bookmark.query
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

    def search_bookmarks(self, query: str) -> List[Dict]:
        if not self.is_ready:
            raise RuntimeError("Model is not initialized")

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

        all_bookmarks = Bookmark.query.all()
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

    def get_topics(self) -> List[Dict]:
        topics = db.session.query(Bookmark.topic, func.count(Bookmark.id)).group_by(Bookmark.topic).all()
        return [{"topic": topic, "count": count} for topic, count in topics]

    def get_visualization_data(self) -> Dict:
        bookmarks = Bookmark.query.all()
        
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

def initialize_model_async(organizer):
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

def create_bookmark_organizer():
    organizer = BookmarkOrganizer()
    return organizer