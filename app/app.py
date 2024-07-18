from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from bertopic_wrapper import BERTopicWrapper
from database import Database
from models import Bookmark, Topic

app = Flask(__name__)
api = Api(app, version='1.0', title='Bookmark Sorter API',
          description='A hierarchical bookmark sorting API using BERTopic')

bertopic_wrapper = BERTopicWrapper()
db = Database()

# Define namespaces
ns_bookmarks = api.namespace('bookmarks', description='Bookmark operations')
ns_topics = api.namespace('topics', description='Topic operations')

# Define models for request/response
bookmark_model = api.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

bookmark_response = api.model('BookmarkResponse', {
    'status': fields.String(description='Operation status'),
    'bookmark_id': fields.Integer(description='ID of the created bookmark'),
    'topics': fields.List(fields.String, description='Assigned topics')
})

@ns_bookmarks.route('/')
class BookmarkList(Resource):
    @api.expect(bookmark_model)
    @api.marshal_with(bookmark_response)
    def post(self):
        """Add a new bookmark"""
        data = request.json
        url = data.get('url')
        title = data.get('title')
        
        # Add bookmark to database
        bookmark_id = db.add_bookmark(url, title)
        
        # Process the bookmark with BERTopic
        topics = bertopic_wrapper.process_bookmark(url, title)
        
        # Update database with topics
        db.update_bookmark_topics(bookmark_id, topics)
        
        return {"status": "success", "bookmark_id": bookmark_id, "topics": topics}

    @api.doc('list_bookmarks')
    def get(self):
        """List all bookmarks"""
        bookmarks = db.get_all_bookmarks()
        return jsonify([b.__dict__ for b in bookmarks])

@ns_topics.route('/hierarchy')
class TopicHierarchy(Resource):
    @api.doc('get_topic_hierarchy')
    def get(self):
        """Get the topic hierarchy"""
        hierarchy = bertopic_wrapper.get_topic_hierarchy()
        return jsonify(hierarchy)

if __name__ == '__main__':
    app.run(debug=True)