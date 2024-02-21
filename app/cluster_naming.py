from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def generate_cluster_names(texts, clusters, num_clusters):
    cluster_groups = {i: [] for i in range(num_clusters)}
    for text, cluster_id in zip(texts, clusters):
        cluster_groups[cluster_id].append(text)

    cluster_names = {}
    for cluster_id, texts_in_cluster in cluster_groups.items():
        prompt = f"Generate a name for a folder of bookmarks with the following contents: {'; '.join(texts_in_cluster)}"
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tool that names clusters of folders, subfolders, and bookmarks. These names should be short and descriptive."},
                {"role": "user", "content": prompt}
            ])
        cluster_names[cluster_id] = completion.choices[0].message.content.strip()
    
    cluster_info = []
    for text, cluster_id in zip(texts, clusters):
        cluster_info.append({"text": text, "cluster": int(cluster_id), "cluster_name": cluster_names[cluster_id]})
    
    return cluster_info
