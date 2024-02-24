# build_html.py

def build_html_import(cluster_info, file_path="clustered_bookmarks.html"):
    """
    Generates an HTML file resembling the Netscape Bookmark File Format from clustered bookmarks data.

    Parameters:
    - cluster_info: List of dictionaries containing clustered bookmarks information.
                    Each dictionary must have 'cluster_name', 'cluster', and a list of 'items' (bookmarks).
    - file_path: The path where the HTML file will be saved.
    """
    for cluster in cluster_info:
        print(cluster)  # To inspect the structure
        if 'items' not in cluster:
            print("No 'items' found in cluster:", cluster)
            continue  # Skip this cluster if no items

    print("Starting HTML file generation...")

    html_content = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
    <!-- This is an automatically generated file.
        It will be read and overwritten.
        DO NOT EDIT! -->
    <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
    <meta http-equiv="Content-Security-Policy"
        content="default-src 'self'; script-src 'none'; img-src data: *; object-src 'none'"></meta>
    <TITLE>Bookmarks</TITLE>
    <H1>Bookmarks Menu</H1>
    <DL><p>
    """

    for cluster in cluster_info:
        cluster_name = cluster.get('cluster_name', 'Unnamed Cluster')
        html_content += f'    <DT><H3 ADD_DATE="" LAST_MODIFIED="">{cluster_name}</H3>\n    <DL><p>\n'
        for item in cluster.get('items', []):  # Safely get 'items' or default to empty list
            bookmark_name = item.get('name', 'Unnamed Bookmark')
            bookmark_url = item.get('url', '#')
            html_content += f'        <DT><A HREF="{bookmark_url}" ADD_DATE="" LAST_MODIFIED="" ICON="">{bookmark_name}</A>\n'
        html_content += "    </DL><p>\n"

    html_content += "</DL><p>"



    # Close the main list
    html_content += "</DL><p>"

    print("Finalizing HTML content...")

    # Write the HTML content to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    print(f"HTML file saved: {file_path}")
