from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def generate_cluster_names(bookmarks, clusters, num_clusters):
    cluster_groups = {i: [] for i in range(num_clusters)}
    for bookmark, cluster_id in zip(bookmarks, clusters):
        try:
            bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
            cluster_groups[cluster_id].append(bookmark_description)
        except KeyError as e:
            print(f"Missing key in bookmark: {e}")
            continue  # Skip this bookmark if it's missing required fields

    cluster_names = {}
    for cluster_id, descriptions_in_cluster in cluster_groups.items():
        prompt = f"Generate a name for a folder of bookmarks with the following contents: {'; '.join(descriptions_in_cluster)}"
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates descriptive names for clusters of bookmarks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            cluster_names[cluster_id] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            cluster_names[cluster_id] = "Unnamed Cluster"  # Provide a default name in case of API failure

    cluster_info = []
    for bookmark, cluster_id in zip(bookmarks, clusters):
        try:
            bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
            cluster_info.append({
                "text": bookmark_description,
                "cluster": int(cluster_id),
                "cluster_name": cluster_names.get(cluster_id, "")
            })
        except Exception as e:
            print(f"Error processing bookmark for cluster info: {e}")
            continue  # Skip this bookmark if there's an issue

    return cluster_info
