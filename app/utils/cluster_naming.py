from openai import OpenAI


import logging
import time
import re
import dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  # Replace with your actual API key

def generate_cluster_names(bookmarks, clusters, num_clusters):
    start_time = time.time()

    cluster_groups = {i: [] for i in range(num_clusters)}
    for bookmark, cluster_id in zip(bookmarks, clusters):
        try:
            bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
            cluster_groups[cluster_id].append(bookmark_description)
        except KeyError as e:
            logging.warning(f"Missing key in bookmark: {e}")
            # Skip this bookmark if it's missing required fields

    cluster_names = {}
    for cluster_id, descriptions_in_cluster in cluster_groups.items():
        print(f"Generating name for cluster {cluster_id}...")
        
        # Preprocess the bookmark descriptions
        cleaned_descriptions = [preprocess_description(desc) for desc in descriptions_in_cluster]
        
        prompt = f"Generate a concise and relevant name for a group of bookmarks with the following descriptions:\n\n{', '.join(cleaned_descriptions)}\n\nExample cluster names:\nTechnology News\nTravel Destinations\nRecipes and Cooking\n\nCluster name:"
        
        try:
            # Generate a cluster name using the OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you're using that model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            # Check if the response is not None and has a 'content' attribute
            if response.choices and hasattr(response.choices[0].message, 'content'):
                generated_name = postprocess_name(response.choices[0].message.content.strip())
                # Implement error handling to ensure appropriate names are set for the clusters
                if generated_name:
                    cluster_names[cluster_id] = generated_name
                else:
                    cluster_names[cluster_id] = f"Cluster {cluster_id}"
            else:
                logging.error("Received invalid response from OpenAI API.")
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
                cluster_names[cluster_id] = f"Cluster {cluster_id}"

            print(f"Generated name for cluster {cluster_id}: {cluster_names[cluster_id]}")
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            cluster_names[cluster_id] = f"Cluster {cluster_id}"  # Provide a default name in case of generation failure
            print(f"Error generating name for cluster {cluster_id}. Using default name: {cluster_names[cluster_id]}")

    cluster_info = []
    for bookmark, cluster_id in zip(bookmarks, clusters):
        try:
            bookmark_description = f"{bookmark['name']} - {bookmark['url']}"
            cluster_info.append({
                "text": bookmark_description,
                "cluster": int(cluster_id),
                "cluster_name": cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            })
        except Exception as e:
            logging.error(f"Error processing bookmark for cluster info: {e}")
            # Skip this bookmark if there's an issue

    end_time = time.time()
    print(f"Total time taken for generating cluster names: {end_time - start_time} seconds")

    return cluster_info

def preprocess_description(description):
    # Remove URLs and special characters from the description
    cleaned_description = re.sub(r'http\S+', '', description)
    cleaned_description = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_description)
    
    # Truncate long descriptions
    max_length = 100
    if len(cleaned_description) > max_length:
        cleaned_description = cleaned_description[:max_length] + "..."
    
    return cleaned_description

def postprocess_name(name):
    # Remove any irrelevant or incoherent parts from the generated name
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    name = name.strip()
    
    return name