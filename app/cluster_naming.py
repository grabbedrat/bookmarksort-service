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

        # Initialize messages with the system instructions
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates descriptive names for clusters of bookmarks."
            }
        ]
        # print messages count
        

        # Append the user prompt with the current cluster's descriptions
        messages.append({"role": "user", "content": prompt})

        print(len(messages))

        # save to file
        with open('messages.txt', 'w') as f:
            for message in messages:
                f.write(f"{message['role']}: {message['content']}\n")

        # Make the API call with the messages
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the content if it's not None and strip whitespace
        if response.choices[0].message.content is not None:
            cluster_names[cluster_id] = response.choices[0].message.content

    # Create the cluster info with names
    cluster_info = []
    for bookmark, cluster_id in zip(bookmarks, clusters):
        bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
        cluster_info.append({
            "text": bookmark_description,
            "cluster": int(cluster_id),
            "cluster_name": cluster_names.get(cluster_id, "")
        })

    return cluster_info
