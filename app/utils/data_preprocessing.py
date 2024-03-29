import re

def clean_bookmark_data(bookmark_data):
    cleaned_data = []
    for bookmark in bookmark_data:
        cleaned_bookmark = {
            "name": clean_text(bookmark["name"]),
            "url": bookmark["url"],
            "tags": bookmark.get("tags", [])  # Keep existing tags
        }
        cleaned_data.append(cleaned_bookmark)
    return cleaned_data

def clean_text(text):
    # Remove special characters and formatting
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def add_tags_to_bookmarks(bookmark_data):
    tagged_data = []
    for bookmark in bookmark_data:
        existing_tags = bookmark.get("tags", [])
        generated_tags = generate_tags(bookmark["name"], bookmark["url"])
        tags = list(set(existing_tags + generated_tags))  # Merge and remove duplicates
        tagged_bookmark = {
            "name": bookmark["name"],
            "url": bookmark["url"],
            "tags": tags
        }
        tagged_data.append(tagged_bookmark)
    return tagged_data

def generate_tags(name, url):
    # Implement your tag generation logic here
    tags = []
    # Example: Add a tag based on the domain name
    domain = extract_domain(url)
    tags.append(domain)
    return tags

def extract_domain(url):
    # Implement domain extraction logic here
    # Example: Extract the domain name from the URL
    domain = url.split("//")[-1].split("/")[0]
    return domain