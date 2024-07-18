from dataclasses import dataclass
from typing import List

@dataclass
class Bookmark:
    id: int
    url: str
    title: str
    topics: List[str] = None

    def __post_init__(self):
        if isinstance(self.topics, str):
            self.topics = self.topics.split(', ')
        elif self.topics is None:
            self.topics = []

@dataclass
class Topic:
    id: int
    name: str
    parent_id: int = None
    children: List['Topic'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: 'Topic'):
        self.children.append(child)