from sklearn.cluster import KMeans

def k_means_cluster(model, num_clusters):
    X = model.wv[model.wv.vocab]
    return KMeans(n_clusters=num_clusters, random_state=0).fit(X)


def color_nodes(graph, membership, words, cluster2color):
    color_map = [None] * len(graph)
    for node in graph:
        assigned_cluster = membership[words.index(str(node))]
        color_map[int(node) - 1] = cluster2color[assigned_cluster]
    return color_map


def cluster_colors(graph, model, cluster2color, num_clusters=6):
    kmeans = k_means_cluster(model, num_clusters)
    words = list(model.wv.vocab)

    membership = kmeans.labels_
    return color_nodes(graph, membership, words, cluster2color)
