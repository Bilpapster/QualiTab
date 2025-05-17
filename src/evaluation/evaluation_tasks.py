def linear_probing(embeddings, labels):
    """
    Performs linear probing on the embeddings. Returns the predicted probabilities for each class.
    """
    from sklearn.linear_model import LogisticRegression

    # Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, labels)

    # Make predictions on the test set and return the predictions
    y_pred_proba = clf.predict_proba(embeddings)
    return y_pred_proba

def k_means_clustering(embeddings, n_clusters, random_state=42):
    """
    Performs K-means clustering on the embeddings.
    This function clusters the embeddings into n_clusters and returns the cluster labels.
    """
    from sklearn.cluster import KMeans

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    return kmeans.fit_predict(embeddings)
