def process_json_data(json_data):
    """
    Process JSON data to extract relevant information for each bookmark.
    
    :param json_data: List of dictionaries representing the simplified bookmark structure,
                      with each containing 'name', 'url', and 'tags'.
    :return: List of dictionaries with 'name', 'url', and 'tags' for each bookmark.
    """
    # Initialize an empty list to hold the processed data
    results = []
    
    # Iterate through each item in the provided JSON data
    for item in json_data:
        # create a heirchical json where each unique "cluster" is a folder with "cluster_name" as the name and bookmarks as children with name and url pulled out of the text which is delimetered by " - "
        for bookmark in item['bookmarks']:
            # Extract the relevant information for each bookmark
            name = bookmark['name']
            url = bookmark['url']
            tags = bookmark['tags']
            
            # Add the extracted data to the results list
            results.append({
                "name": name,
                "url": url,
                "tags": tags
            })
    
    return results
