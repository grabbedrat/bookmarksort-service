import re
import openai
import os
from multiprocessing import Pool

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

def generate_tags_batch(batch):
    descriptions = [f"URL: {bookmark['url']}\nTitle: {bookmark['name']}\n" for bookmark in batch]
    prompt = f"Generate 6 relevant tags for each of the following bookmarks based on the provided URL and title. " \
             f"Format the tags as a comma-separated list, with each tag surrounded by square brackets, like this: " \
             f"'[tag1], [tag2], [tag3], [tag4], [tag5], [tag6]'. Each bookmark's tags should be on a separate line. " \
             f"Do not include any numbers or additional text before the tags. Focus on generating tags that capture " \
             f"the main topics, categories, key information, and any other relevant details.\n\n" \
             f"{''.join(descriptions)}\nTags:"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates comprehensive and relevant tags for bookmarks. "
                                              "Follow the specified format for the tags: a comma-separated list with each tag surrounded by square brackets, "
                                              "like this: '[tag1], [tag2], [tag3], [tag4], [tag5], [tag6]'. Each bookmark's tags should be on a separate line. "
                                              "Do not include any numbers or additional text before the tags. Consistency in formatting is essential for proper processing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )
        if response.choices:
            generated_tags = response.choices[0].message.content.strip()
        else:
            generated_tags = ""
        extracted_tags = []
        for line in generated_tags.split("\n"):
            if line.strip():
                tags = re.findall(r'\[(.*?)\]', line)
                extracted_tags.append(tags)
        return extracted_tags
    except Exception as e:
        print(f"Error generating tags: {e}")
        return [[] for _ in batch]  # Return empty tags for each bookmark in case of an error

def process_batch(batch):
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            generated_tags = generate_tags_batch(batch)
            tagged_batch = []
            for j, bookmark in enumerate(batch):
                existing_tags = bookmark.get("tags", [])
                tags = list(set(existing_tags + generated_tags[j]))  # Merge and remove duplicates
                tagged_bookmark = {
                    "name": bookmark["name"],
                    "url": bookmark["url"],
                    "tags": tags
                }
                tagged_batch.append(tagged_bookmark)
            return tagged_batch
        except Exception as e:
            print(f"Error processing batch: {e}")
            retries += 1
            if retries == max_retries:
                print(f"Max retries reached for batch. Skipping batch.")
                return []

def add_tags_to_bookmarks(bookmark_data, batch_size=10, num_processes=8):
    batches = [bookmark_data[i:i+batch_size] for i in range(0, len(bookmark_data), batch_size)]
    
    with Pool(processes=num_processes) as pool:
        tagged_data = pool.map(process_batch, batches)
    
    tagged_data = [item for sublist in tagged_data for item in sublist]  # Flatten the list of lists
    return tagged_data