def build_structured_json(clusters, cluster_info):
    structured_data = []

    def create_folder(cluster_id, parent_id=None):
        folder = {
            "type": "folder",
            "id": int(cluster_id),
            "name": cluster_info[cluster_id],
            "children": []
        }

        if parent_id is not None:
            folder["parent_id"] = int(parent_id)

        for bookmark in clusters[cluster_id]:
            bookmark_name = bookmark["name"].strip()
            if not bookmark_name:
                bookmark_name = "Untitled Bookmark"

            folder["children"].append({
                "type": "bookmark",
                "name": bookmark_name,
                "url": bookmark["url"]
            })

        return folder

    def assign_parent_folders(folder, parent_id, depth=0, max_depth=5):
        if depth > max_depth:
            structured_data.append(folder)
            return

        if parent_id is not None:
            parent_folder = next((f for f in structured_data if f["type"] == "folder" and f.get("id") == parent_id), None)
            if parent_folder:
                parent_folder["children"].append(folder)
            else:
                assign_parent_folders(folder, parent_id // 2, depth + 1)
        else:
            structured_data.append(folder)

    for cluster_id in clusters:
        folder = create_folder(cluster_id)
        parent_id = cluster_id // 2
        assign_parent_folders(folder, parent_id)

    return structured_data