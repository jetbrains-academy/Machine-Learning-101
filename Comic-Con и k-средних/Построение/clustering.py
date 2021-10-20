import numpy as np


def init_clusters(n_clusters, n_features):
    return np.random.randint(low=0, high=255, size=(n_clusters, n_features))


def k_means(X, n_clusters, distance_metric):
    n_samples, n_features = X.shape
    classification = np.zeros(n_samples)
    clusters = init_clusters(n_clusters, n_features)
    distance = np.zeros((n_clusters, n_samples))

    while True:
        for i, c in enumerate(clusters):
            distance[i] = distance_metric(X, c)
        new_classification = np.argmin(distance, axis=0)
        if np.sum(new_classification != classification) == 0:
            break
        classification = new_classification
        for i in range(n_clusters):
            mask = classification == i
            total_classified = np.sum(mask)
            is_empty_cluster = total_classified == 0
            if not is_empty_cluster:
                clusters[i] = np.sum(X[mask], axis=0) / total_classified
            else:
                clusters[i] = X[np.argmax(distance[i])]
    return classification, clusters
