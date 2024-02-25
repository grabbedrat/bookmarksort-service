# bookmark_import.py

def build_json_import(cluster_info):
    """

    """
    # Initialize a dictionary to hold the clusters
    clusters_dict = {}

    for item in cluster_info:
        # Parse the 'text' to separate the name and URL using the ' - ' delimiter (using last instance of ' - ' to allow for URLs with ' - ' in them)
        name, url = item["text"].rsplit(" - ", 1)

        # Create a new entry for each cluster if it doesn't already exist
        if item["cluster_name"] not in clusters_dict:
            clusters_dict[item["cluster_name"]] = []

        # Append the bookmark to the appropriate cluster
        clusters_dict[item["cluster_name"]].append({
            "name": name,
            "url": url
        })

    return clusters_dict