from flask import Flask
from flask_restx import Api, Resource, fields
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from dataclasses import dataclass, asdict, field
from typing import List
import logging

app = Flask(__name__)
api = Api(app, version='1.0', title='Bookmark Sorting API',
          description='An API for sorting and categorizing bookmarks using BERTopic')

logging.basicConfig(level=logging.INFO)

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
        self.model = BERTopic(
            language="english",
            min_topic_size=5,
            nr_topics="auto",
            calculate_probabilities=True,
            verbose=True
        )
        self.vectorizer = CountVectorizer(stop_words="english")
        self.documents = []

    def process_bookmarks(self, bookmarks):
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

topic_wrapper = BERTopicWrapper()

bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title'),
    'tags': fields.List(fields.String, description='List of tags associated with the bookmark')
})

bookmarks_model = api.model('BookmarkList', {
    'bookmarks': fields.List(fields.Nested(bookmark_model), required=True, description='List of bookmarks to process')
})

bookmark_result_model = api.model('BookmarkResult', {
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'tags': fields.List(fields.String, description='List of tags associated with the bookmark'),
    'topics': fields.List(fields.String, description='List of topics assigned to the bookmark'),
    'probabilities': fields.List(fields.Float, description='Probabilities for each topic')
})

@ns.route('/')
class BookmarksResource(Resource):
    @ns.expect(bookmarks_model)
    @ns.marshal_list_with(bookmark_result_model)
    def post(self):
        """Process a list of bookmarks and assign topics"""
        bookmarks_data = api.payload['bookmarks']
        bookmarks = [Bookmark(url=b['url'], title=b['title'], tags=b.get('tags', [])) for b in bookmarks_data]
        results = topic_wrapper.process_bookmarks(bookmarks)
        return [asdict(result) for result in results]

if __name__ == '__main__':
    app.run(debug=True)