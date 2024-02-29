import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

def plot_clusters(embeddings, clusters, num_clusters):
    n_samples = len(embeddings)
    perplexity_value = min(30, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    print(plt.colormaps())

    plt.figure(figsize=(10, 8))
    # Define colors using a colormap which creates an array of colors based on the number of clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))  # Ensure 'rainbow' colormap is called from plt.cm

    for i in range(num_clusters):
        idx = clusters == i
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], color=colors[i], label=f"Cluster {i+1}", alpha=0.6)

    plt.title('Clusters of Bookmarks')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.legend()

    # Save the figure
    plt.savefig('clustered_bookmarks.png', dpi=300)
    plt.close()
