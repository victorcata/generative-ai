from sklearn.neighbors import NearestNeighbors


def build_search_index(embeddings, df, n_neighbors=5):
    """
    Build a search index using NearestNeighbors.
    :param embeddings: List of embeddings to build the index from.
    :param n_neighbors: Number of neighbors to find.
    :param algorithm: Algorithm to use for nearest neighbors search.
    :return: NearestNeighbors instance, distances, and indices of neighbors.
    """

    # Create the search index
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm='ball_tree').fit(embeddings)

    # To query the index, you can use the kneighbors method
    distances, indices = nbrs.kneighbors(embeddings)

    # Store the indices and distances in the DataFrame
    df['indices'] = indices.tolist()
    df['distances'] = distances.tolist()

    return nbrs
