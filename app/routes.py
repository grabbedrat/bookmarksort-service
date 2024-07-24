# routes.py
from flask import request, current_app
from flask_restx import Namespace, Resource, fields
from http import HTTPStatus
from models import Bookmark
from flask import jsonify

ns = Namespace('bookmarks', description='Bookmark operations')

# API models
bookmark_model = ns.model('Bookmark', {
    'url': fields.String(required=True, description='The bookmark URL'),
    'title': fields.String(required=True, description='The bookmark title')
})

bookmark_response_model = ns.model('BookmarkResponse', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'organized_bookmarks': fields.Raw(description='Organized bookmarks by topic')
})

bookmarks_list_model = ns.model('BookmarksList', {
    'bookmarks': fields.List(fields.Nested(ns.model('BookmarkDetail', {
        'url': fields.String(description='The bookmark URL'),
        'title': fields.String(description='The bookmark title'),
        'topic': fields.String(description='The bookmark topic')
    }))),
    'total': fields.Integer(description='Total number of bookmarks'),
    'page': fields.Integer(description='Current page number'),
    'per_page': fields.Integer(description='Number of bookmarks per page'),
    'total_pages': fields.Integer(description='Total number of pages')
})

search_result_model = ns.model('SearchResult', {
    'url': fields.String(description='The bookmark URL'),
    'title': fields.String(description='The bookmark title'),
    'topic': fields.String(description='The bookmark topic'),
    'similarity': fields.Float(description='Similarity score')
})

topic_model = ns.model('Topic', {
    'topic': fields.String(description='Topic name'),
    'count': fields.Integer(description='Number of bookmarks in this topic')
})

visualization_model = ns.model('Visualization', {
    'success': fields.Boolean(description='Whether the operation was successful'),
    'visualization_data': fields.Raw(description='Visualization data for bookmarks')
})

update_params_model = ns.model('UpdateParams', {
    'embedding_model': fields.String(description='Embedding model name'),
    'umap_n_neighbors': fields.Integer(description='UMAP n_neighbors parameter'),
    'umap_n_components': fields.Integer(description='UMAP n_components parameter'),
    'umap_min_dist': fields.Float(description='UMAP min_dist parameter'),
    'hdbscan_min_cluster_size': fields.Integer(description='HDBSCAN min_cluster_size parameter'),
    'hdbscan_min_samples': fields.Integer(description='HDBSCAN min_samples parameter'),
    'nr_topics': fields.String(description='Number of topics'),
    'top_n_words': fields.Integer(description='Number of top words per topic')
})

def init_routes(api):
    api.add_namespace(ns)

    @ns.route('/process')
    class ProcessBookmarks(Resource):
        @ns.marshal_with(bookmark_response_model)
        def post(self):
            """Process all bookmarks in the database and organize them by topics"""
            organizer = current_app.organizer
            if not organizer.is_ready:
                return {"success": False, "error": "Model is still initializing. Please try again later."}, HTTPStatus.SERVICE_UNAVAILABLE

            try:
                result = organizer.process_bookmarks()
                return {"success": True, "organized_bookmarks": result}
            except Exception as e:
                current_app.logger.error(f"Error processing bookmarks: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

    @ns.route('/add')
    class AddBookmark(Resource):
        @ns.expect(bookmark_model)
        @ns.marshal_with(bookmark_response_model)
        def post(self):
            """Add a new bookmark"""
            organizer = current_app.organizer
            if not organizer.is_ready:
                return {"success": False, "error": "Model is still initializing. Please try again later."}, HTTPStatus.SERVICE_UNAVAILABLE

            bookmark = request.json
            try:
                result = organizer.add_bookmark(bookmark)
                return {"success": True, "organized_bookmarks": result}
            except Exception as e:
                current_app.logger.error(f"Error adding bookmark: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR

    @ns.route('/list')
    @ns.param('topic', 'Filter bookmarks by topic (optional)')
    @ns.param('page', 'Page number (default: 1)')
    @ns.param('per_page', 'Number of bookmarks per page (default: 20)')
    class ListBookmarks(Resource):
        @ns.marshal_with(bookmarks_list_model)
        def get(self):
            """List all bookmarks, optionally filtered by topic"""
            organizer = current_app.organizer
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
            organizer = current_app.organizer
            if not organizer.is_ready:
                return {"error": "Model is still initializing. Please try again later."}, HTTPStatus.SERVICE_UNAVAILABLE

            query = request.args.get('q')
            if not query:
                return {"error": "Search query is required"}, HTTPStatus.BAD_REQUEST

            return organizer.search_bookmarks(query)

    @ns.route('/topics')
    class Topics(Resource):
        @ns.marshal_with(topic_model)
        def get(self):
            """Get all topics and their bookmark counts"""
            organizer = current_app.organizer
            return organizer.get_topics()

    @ns.route('/visualization')
    class Visualization(Resource):
        @ns.marshal_with(visualization_model)
        def get(self):
            """Get visualization data for bookmarks"""
            organizer = current_app.organizer
            if not organizer.is_ready:
                return {"success": False, "error": "Model is still initializing. Please try again later."}, HTTPStatus.SERVICE_UNAVAILABLE
            
            try:
                visualization_data = organizer.get_visualization_data()
                return {"success": True, "visualization_data": visualization_data}
            except Exception as e:
                current_app.logger.error(f"Error getting visualization data: {str(e)}")
                return {"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR
            
    @ns.route('/update_params')
    class UpdateParams(Resource):
        @ns.expect(update_params_model)
        def post(self):
            """Update the parameters for the bookmark organizer"""
            organizer = current_app.organizer
            new_params = request.json
            try:
                organizer.update_parameters(new_params)
                return jsonify({"success": True, "message": "Parameters updated successfully"})
            except Exception as e:
                current_app.logger.error(f"Error updating parameters: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR