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
    for cluster_id, descriptions_in_cluster in cluster_groups.items():
        prompt = f"Generate a name for a folder of bookmarks with the following contents: {'; '.join(descriptions_in_cluster)}"
        
        # Reinitialize messages at the start of each iteration with the correct structure
        messages = [
        {
            "role": "system",
            "content": "You are a tool that names clusters of folders, subfolders, and bookmarks. These names should be short and descriptive."
        },
        {
            "role": "user",
            "content": "Generate a name for a folder of bookmarks with the following contents: ..."
        }
        # You can add more messages as needed
    ]
        
        completions = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
        {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
        {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
        {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
        {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ] # Ensure this addition matches the expected structure
        )
        
        # Assuming you want to get a single completion per cluster and then reset messages for the next iteration
        if completions.choices[0].message.content is not None:
            cluster_names[cluster_id] = completions.choices[0].message.content.strip()
        messages.pop()  # Remove the last user message to prepare for the next iteration

    cluster_info = []
    for bookmark, cluster_id in zip(bookmarks, clusters):
        bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
        cluster_info.append({
            "text": bookmark_description, 
            "cluster": int(cluster_id), 
            "cluster_name": cluster_names.get(cluster_id, "")  # Use .get to avoid KeyError
        })

    return cluster_info
