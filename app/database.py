import sqlite3
from models import Bookmark, Topic

class Database:
    def __init__(self, db_name='bookmarks.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            bookmark_id INTEGER,
            topic_name TEXT NOT NULL,
            FOREIGN KEY (bookmark_id) REFERENCES bookmarks (id)
        )
        ''')
        self.conn.commit()

    def add_bookmark(self, url, title):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO bookmarks (url, title) VALUES (?, ?)', (url, title))
        self.conn.commit()
        return cursor.lastrowid

    def update_bookmark_topics(self, bookmark_id, topics):
        cursor = self.conn.cursor()
        for topic in topics:
            cursor.execute('INSERT INTO topics (bookmark_id, topic_name) VALUES (?, ?)', 
                           (bookmark_id, topic))
        self.conn.commit()

    def get_all_bookmarks(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT b.id, b.url, b.title, GROUP_CONCAT(t.topic_name, ', ') as topics
        FROM bookmarks b
        LEFT JOIN topics t ON b.id = t.bookmark_id
        GROUP BY b.id
        ''')
        return [Bookmark(*row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()