from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class BERTopicWrapper:
    def __init__(self):
        # Initialize BERTopic model
        self.model = BERTopic(language="english", 
                              min_topic_size=5, 
                              nr_topics="auto")
        self.vectorizer = CountVectorizer(stop_words="english")
        self.documents = []
        self.topics = []

    def process_bookmark(self, url, title):
        # Combine URL and title for processing
        document = f"{title} {url}"
        self.documents.append(document)
        
        # Fit the model if it's the first document, otherwise transform
        if len(self.documents) == 1:
            self.topics, _ = self.model.fit_transform(self.documents)
        else:
            new_topic, _ = self.model.transform([document])
            self.topics.extend(new_topic)
        
        # Return the assigned topic(s) for the new bookmark
        return self.model.get_topic(self.topics[-1])

    def get_topic_hierarchy(self):
        # Get hierarchical topics
        hierarchical_topics = self.model.hierarchical_topics(self.documents)
        
        # Convert to a more suitable format for frontend consumption
        hierarchy = {}
        for level in hierarchical_topics:
            hierarchy[f"Level_{level}"] = hierarchical_topics[level]
        
        return hierarchy

    def update_model(self):
        # Retrain the model with all documents
        self.topics, _ = self.model.fit_transform(self.documents)