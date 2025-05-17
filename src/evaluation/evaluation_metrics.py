def purity(labels_true, labels_pred):
    """Calculates the purity score for clustering."""
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances, normalize

    from collections import Counter
    cluster_purity = []
    for cluster_id in np.unique(labels_pred):
        cluster_labels = labels_true[labels_pred == cluster_id]
        most_common = Counter(cluster_labels).most_common(1)
        purity = most_common[0][1] / len(cluster_labels) if len(cluster_labels) > 0 else 0
        cluster_purity.append(purity)
    return np.mean(cluster_purity)


def knn_avg_cosine_similarity(embeddings, k_values):
    """Calculates the average cosine similarity to the k-nearest neighbors."""

    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    knn_results = {}
    for k in k_values:
        if len(embeddings) < k + 1:
            print(f"Warning: Not enough samples for k={k}. Skipping.")
            continue
        knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')  # +1 to exclude self
        knn.fit(embeddings)
        distances, _ = knn.kneighbors(embeddings)
        # Average distance to the k nearest neighbors (excluding self)
        avg_similarity = np.mean(1 - distances[:, 1:])  # Convert distance to similarity
        knn_results[f'Avg Cosine Similarity (k={k})'] = avg_similarity
    return knn_results


def silhouette_score(embeddings, labels):
    """Calculates the silhouette score for clustering."""
    from sklearn.metrics import silhouette_score
    import numpy as np

    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(embeddings, labels)


def z_normalize(embeddings):
    """Normalizes the embeddings to have zero mean and unit variance. Normalizes each row separately."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings.T).T  # Transpose to normalize every row
    return normalized_embeddings


def euclidean_distances_from_reference(reference_embeddings, target_embeddings, z_normalize_first=True):
    """
    Calculates the Euclidean distance from each target embedding to a reference embedding.
    For instance, calculates the distance between the first target embedding and the first reference embedding,
    the second target embedding and the second reference embedding, etc. The result is a 1D array of distances.

    Args:
        reference_embeddings (np.ndarray): Reference embeddings of shape (n_samples, n_features).
        target_embeddings (np.ndarray): Target embeddings of shape (n_samples, n_features).
        z_normalize_first (bool): If True, z-normalize the embeddings before calculating Euclidean distances.
    """
    import math
    import numpy as np

    if z_normalize_first:
        reference_embeddings = z_normalize(reference_embeddings)
        target_embeddings = z_normalize(target_embeddings)

    return np.array([
        math.dist(reference_embedding, target_embedding)
        for reference_embedding, target_embedding in zip(reference_embeddings, target_embeddings)
    ])


def cosine_similarity_from_reference(reference_embeddings, target_embeddings):
    """
    Calculates the cosine similarity from each target embedding to a reference embedding.
    For instance, calculates the similarity between the first target embedding and the first reference embedding,
    the second target embedding and the second reference embedding, etc. The result is a 1D array of similarities.

    Args:
        reference_embeddings (np.ndarray): Reference embeddings of shape (n_samples, n_features).
        target_embeddings (np.ndarray): Target embeddings of shape (n_samples, n_features).
    """
    from scipy.spatial.distance import cosine

    return [
        cosine(reference_embedding, target_embedding)
        for reference_embedding, target_embedding in zip(reference_embeddings, target_embeddings)
    ]


def roc_auc(y_true, y_predict_proba):
    """Calculates the ROC AUC score. Automatically handles binary and multiclass cases (ovr)."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_predict_proba, multi_class='ovr') if len(y_true) > 2 \
        else roc_auc_score(y_true, y_predict_proba[:, 1])
