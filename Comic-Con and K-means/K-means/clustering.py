import numpy as np


def init_clusters(n_clusters, n_features):
    # Here, we initialize n clusters for a sample of objects with n features.
    # The clusters are generated with random values within the valid range.
    # In this case, the values range from 0 to 255 for the RGB matrix.
    return np.random.randint(low=0, high=255, size=(n_clusters, n_features))


def k_means(X, n_clusters, distance_metric):
    # The sample matrix is organized with objects as rows and features as columns.
    # Therefore, the matrix dimensions correspond to the number of samples
    # and the number of features, respectively.
    n_samples, n_features = X.shape
    # We initialize the classification array with zeros
    # to represent that each sample is currently unaffiliated with a class.
    classification = np.zeros(n_samples)
    # Initializing the target number of clusters
    clusters = init_clusters(n_clusters, n_features)
    # We set the initial distance for each sample
    # to zero until the actual distances to the clusters are computed.
    distance = np.zeros((n_clusters, n_samples))

    # The algorithm will iterate until a stopping condition is reached.
    while True:
        # We iterate through each cluster and calculate its distance to
        # every sample in the dataset.
        for i, c in enumerate(clusters):
            distance[i] = distance_metric(X, c)
        # Samples are assigned to the class of their nearest
        # cluster.
        # The new_classification array stores these
        # updated assignments.
        new_classification = np.argmin(distance, axis=0)
        # The first stop condition is met when the new_classification
        # is identical to the previous one.
        if np.sum(new_classification != classification) == 0:
            break
        classification = new_classification
        # The following loop adjusts cluster centers
        # to the mean position of their members.
        # E.g., if a cluster contains two samples – (0, 0, 0) and (2, 2, 2) –
        # its new center will be at (1, 1, 1).
        for i in range(n_clusters):
            mask = classification == i
            total_classified = np.sum(mask)
            # Here we also check for empty clusters
            # that have no assigned samples.
            # These can cause numerical errors
            # and prevent convergence.
            # If an empty cluster occurs, we reassign its center
            # to the most distant sample in the dataset.
            is_empty_cluster = total_classified == 0
            if not is_empty_cluster:
                clusters[i] = np.sum(X[mask], axis=0) / total_classified
            else:
                clusters[i] = X[np.argmax(distance[i])]
    return classification, clusters