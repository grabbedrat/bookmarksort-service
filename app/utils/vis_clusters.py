import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx


def plot_clusters(cluster_points, color_map):
    plt.switch_backend('Agg')  # Switch to non-interactive backend
    for i, points in enumerate(cluster_points):
        plt.scatter(points[:, 0], points[:, 1], color=color_map[i], alpha=0.5)

def visualize_clusters(clusters, num_clusters, embeddings, output_file):
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))
    color_map = {i: colors[i] for i in range(num_clusters)}

    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    cluster_points = []
    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            cluster_points.append(reduced_embeddings[cluster_indices])

    plot_clusters(cluster_points, color_map)

    plt.title('Hierarchical Clustering Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Example usage:
# visualize_clusters(clusters_array, number_of_clusters, embeddings_array, 'output.png')

def plot_dendrogram(link_matrix, output_file):
    plt.figure(figsize=(10, 7))
    dendrogram(link_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def visualize_structured_data(structured_data, output_file):
    G = nx.DiGraph()

    def add_nodes_recursively(data, parent=None):
        for item in data:
            if item["type"] == "folder":
                folder_name = f"{item['name']} ({len(item['children'])})"
                G.add_node(item["id"], name=folder_name, type="folder")
                if parent is not None:
                    G.add_edge(parent, item["id"])
                add_nodes_recursively(item["children"], parent=item["id"])

    add_nodes_recursively(structured_data)

    labels = nx.get_node_attributes(G, 'name')
    node_colors = ["lightblue" for _ in G.nodes()]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, labels=labels, node_color=node_colors, font_size=10, node_size=1000)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, format="jpeg", dpi=300)
    plt.close()
