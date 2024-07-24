from flask import Flask
from flask_restx import Api, Resource
from flask_cors import CORS
from config import Config
from models import init_db
from routes import init_routes
from bookmark_organizer import create_bookmark_organizer
from threading import Thread
import traceback

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, resources={r"/*": {"origins": "*"}})
    api = Api(app, version='1.0', title='Bookmark Organizer API',
              description='An API for organizing bookmarks using BERTopic')

    init_db(app)
    init_routes(api)

    app.organizer = create_bookmark_organizer()

    def initialize_model_async():
        app.organizer.initialize(
            embedding_model="all-MiniLM-L6-v2",
            umap_n_neighbors=15,
            umap_n_components=5,
            umap_min_dist=0.0,
            hdbscan_min_cluster_size=15,
            hdbscan_min_samples=10,
            nr_topics="auto",
            top_n_words=10
        )

    # Start the initialization in a separate thread
    Thread(target=initialize_model_async).start()

    # Add status endpoint to the API
    @api.route('/status')
    class Status(Resource):
        def get(self):
            try:
                app.logger.info("Status endpoint called")
                if not hasattr(app, 'organizer'):
                    app.logger.error("app.organizer not found")
                    return {"status": "error", "message": "Organizer not initialized"}, 500
                
                app.logger.info(f"Organizer status: is_ready={app.organizer.is_ready}, is_initializing={app.organizer.is_initializing}")
                
                if app.organizer.is_ready:
                    return {"status": "ready"}, 200
                elif app.organizer.is_initializing:
                    return {"status": "initializing"}, 202
                else:
                    return {"status": "not started"}, 500
            except Exception as e:
                app.logger.error(f"Error in status endpoint: {str(e)}")
                app.logger.error(f"Error traceback: {traceback.format_exc()}")
                return {"status": "error", "message": str(e)}, 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, threaded=True)