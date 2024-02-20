from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Original example URLs
texts = list(set([
    "https://www.google.com", "https://www.facebook.com", "https://www.youtube.com", "https://www.yahoo.com",
    "https://www.wikipedia.org", "https://www.amazon.com", "https://www.twitter.com", "https://www.instagram.com",
    "https://www.linkedin.com", "https://www.reddit.com", "https://www.pinterest.com", "https://www.tumblr.com",
    "https://www.microsoft.com", "https://www.apple.com", "https://www.netflix.com", "https://www.spotify.com",
    "https://www.paypal.com", "https://www.ebay.com", "https://www.aliexpress.com", "https://www.alibaba.com",
    "https://www.booking.com", "https://www.airbnb.com", "https://www.tripadvisor.com", "https://www.expedia.com",
    "https://www.cnn.com", "https://www.bbc.com", "https://www.nytimes.com", "https://www.theguardian.com",
    "https://www.washingtonpost.com", "https://www.usatoday.com", "https://www.wsj.com", "https://www.latimes.com",
    "https://www.foxnews.com", "https://www.nbcnews.com", "https://www.cbsnews.com", "https://www.abcnews.go.com",
    "https://www.huffpost.com", "https://www.buzzfeed.com", "https://www.msnbc.com", "https://www.cbs.com",
    "https://www.abc.com", "https://www.nbc.com", "https://www.fox.com", "https://www.cw.com", "https://www.cwtv.com",
    "https://www.cbs.com", "https://www.abc.com", "https://www.nbc.com", "https://www.fox.com", "https://www.cw.com",
    "https://www.cwtv.com", "https://www.netflix.com", "https://www.hulu.com", "https://www.disneyplus.com",
    "https://www.amazon.com", "https://www.primevideo.com", "https://www.apple.com", "https://www.tv.apple.com",
    "https://www.peacocktv.com", "https://www.paramountplus.com", "https://www.hbomax.com", "https://www.discoveryplus.com",
    "https://www.youtube.com", "https://www.youtube.tv", "https://www.twitch.tv", "https://www.vimeo.com",
]))

# Trim URLs to show only the domain name
trimmed_texts = [text.split("//")[-1].split(".")[1] for text in texts]

# Generate embeddings
embeddings = model.encode(texts)

# Cluster embeddings
num_clusters = 8
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Determine an appropriate perplexity value
n_samples = len(texts)
perplexity_value = min(30, max(5, n_samples // 3))

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
reduced_embeddings = tsne.fit_transform(embeddings)

# Visualization
plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

for i, (x, y) in enumerate(reduced_embeddings):
    plt.scatter(x, y, color=colors[clusters[i]])
    plt.text(x, y, trimmed_texts[i], fontsize=9)

plt.title('Text Clusters visualized with t-SNE')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')

# Save and display the plot
plt.savefig('clusters_trimmed_no_legend.png')
plt.show()
