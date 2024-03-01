from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, origins=["moz-extension://3536d120-2cde-4eb9-b29c-bcdabc94d9c4"])

    with app.app_context():
        from . import routes
        return app
