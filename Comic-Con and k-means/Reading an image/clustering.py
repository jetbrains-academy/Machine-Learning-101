import numpy as np


def init_clusters(n_clusters, n_features):
    # Here we create n initial clusters for a sample of objects with n features.
    # The clusters are generated randomly from the acceptable values.
    # In our case, those are 0..255 for the RGB matrix.
    return np.random.randint(low=0, high=255, size=(n_clusters, n_features))


def k_means(X, n_clusters, distance_metric):
    # The sample matrix has objects along one of its sides and features along the other,
    # so the number of provided samples is equal to one dimension of the matrix,
    # while the number of features is equal to the other.
    n_samples, n_features = X.shape
    # The initial classification for each of the samples is a zero
    # array, as they are not yet affiliated with a class.
    classification = np.zeros(n_samples)
    # Initializing the requested amount of clusters to adjust
    clusters = init_clusters(n_clusters, n_features)
    # For each of the samples, we set the initial distance to all of the
    # clusters to zero, as the real distances are not yet measured.
    distance = np.zeros((n_clusters, n_samples))

    # The algorithm will iterate until a stop condition is met.
    while True:
        # We enumerate the clusters and calculate the distances to them
        # from all of the samples.
        for i, c in enumerate(clusters):
            distance[i] = #TODO
        # The assigned classes would be those corresponding to the nearest
        # cluster for each of the samples.
        # The new_classification is an array storing all of the
        # classes assigned.
        new_classification = #TODO
        # The first stop condition would be met if the new_classification
        # is exactly the same as the previous classification.
        if #TODO:
            break
        classification = new_classification
        # The following loop adjusts the centers of the clusters
        # based on the mean values of the samples classified.
        # E.g., if a cluster has two objects, (0, 0, 0) and (2, 2, 2),
        # its new center will be at (1, 1, 1).
        for i in range(n_clusters):
            mask = classification == i
            total_classified = np.sum(mask)
            # Here we also check if there appears an empty cluster
            # without any corresponding samples.
            # Those could be dangerous and destabilize the
            # algorithm, resulting in it never finishing its
            # work. In case such clusters occur, we assign them
            # a new center equal to the most distant sample.
            is_empty_cluster = total_classified == 0
            if not is_empty_cluster:
                clusters[i] = #TODO
            else:
                clusters[i] = X[np.argmax(distance[i])]
    return classification, clusters
