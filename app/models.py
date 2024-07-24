# models.py
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, PickleType

db = SQLAlchemy()

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

class Bookmark(db.Model):
    __tablename__ = 'bookmarks'

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    embedding = Column(PickleType, nullable=False)
    topic = Column(String, nullable=False)