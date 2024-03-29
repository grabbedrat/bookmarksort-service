import re
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your actual API key

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

def add_tags_to_bookmarks(bookmark_data, batch_size=5):
    tagged_data = []
    for i in range(0, len(bookmark_data), batch_size):
        batch = bookmark_data[i:i+batch_size]
        generated_tags = generate_tags_batch(batch)
        for j, bookmark in enumerate(batch):
            existing_tags = bookmark.get("tags", [])
            domain_tag = extract_domain(bookmark["url"])
            tags = list(set(existing_tags + generated_tags[j] + [domain_tag]))  # Merge and remove duplicates
            tagged_bookmark = {
                "name": bookmark["name"],
                "url": bookmark["url"],
                "tags": tags
            }
            tagged_data.append(tagged_bookmark)
    return tagged_data

def generate_tags_batch(bookmarks):
    descriptions = [f"URL: {bookmark['url']}\nTitle: {bookmark['name']}\n" for bookmark in bookmarks]
    prompt = f"Generate 4 relevant tags for each of the following bookmarks. Format the tags as a comma-separated list of numbers and tags, like this: '1. tag1, tag2, tag3, tag4'. Focus on generating tags that capture the main topics, categories, key information, and any other relevant details from the URL and title.\n\n{''.join(descriptions)}\nTags:"

    print(f"Prompt:\n{prompt}\n")  # Print the prompt

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates comprehensive and relevant tags for bookmarks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )

        generated_tags = response.choices[0].message.content.strip()
        print(f"Response:\n{generated_tags}\n")  # Print the response

        extracted_tags = []
        for line in generated_tags.split("\n"):
            if line.strip():
                tags = line.split(". ")[1].split(", ")
                extracted_tags.append(tags)

        print(f"Extracted Tags:\n{extracted_tags}\n")  # Print the extracted tags

        return extracted_tags
    except Exception as e:
        print(f"Error generating tags: {e}")
        return [[] for _ in bookmarks]  # Return empty tags for each bookmark in case of an error

def extract_domain(url):
    # Extract the domain name from the URL
    domain = url.split("//")[-1].split("/")[0]
    return domain