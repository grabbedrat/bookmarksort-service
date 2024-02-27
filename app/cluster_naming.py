from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def generate_cluster_names(bookmarks, clusters, num_clusters):
    cluster_groups = {i: [] for i in range(num_clusters)}
    for bookmark, cluster_id in zip(bookmarks, clusters):
        bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
        cluster_groups[cluster_id].append(bookmark_description)

    cluster_names = {}
    messages = [{"role": "system", "content": "You are a tool that names clusters of folders, subfolders, and bookmarks. These names should be short and descriptive."}]
    for cluster_id, descriptions_in_cluster in cluster_groups.items():
        prompt = f"Generate a name for a folder of bookmarks with the following contents: {'; '.join(descriptions_in_cluster)}"
        messages.append({"role": "user", "content": prompt})

    completions = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    for i, completion in enumerate(completions.choices):
        cluster_names[i] = completion.message.content.strip()

    cluster_info = []
    for bookmark, cluster_id in zip(bookmarks, clusters):
        bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
        cluster_info.append({"text": bookmark_description, "cluster": int(cluster_id), "cluster_name": cluster_names[cluster_id]})

        return cluster_info
