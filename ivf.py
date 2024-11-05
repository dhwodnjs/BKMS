
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class IVF:

    def _compute_distance(self, data, q):
        """Compute the appropriate distance based on the distance type."""
        if self.distance_type == 'l2':
            return np.linalg.norm(data - q, axis=1)
        elif self.distance_type == 'cosine':
            return 1 - np.dot(data, q.T) / (np.linalg.norm(data, axis=1) * np.linalg.norm(q))


    def __init__(self, distance_type, n_clusters):

        self.n_clusters = n_clusters
        self.distance_type = distance_type
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.inverted_index = {i: [] for i in range(n_clusters)}
        self.data = None


    def fit(self, X):

        if self.distance_type == 'cosine':
            X = X / np.linalg.norm(X, axis=1, keepdims=True)

        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        
        for idx, label in enumerate(labels):
            self.inverted_index[label].append(idx)

        self.data = X



    def search(self, q, k, n_probes=1):

        assert n_probes <= self.n_clusters 

        if self.distance_type == "cosine":
            q = q / np.linalg.norm(q)
        
        cluster_distances = self._compute_distance(self.kmeans.cluster_centers_, q).flatten()    
        probe_clusters = np.argpartition(cluster_distances, n_probes)[:n_probes]

        candidates = np.concatenate([self.inverted_index[cl] for cl in probe_clusters])

        if len(candidates) <= k:
            return candidates
        
        dists = self._compute_distance(self.data[candidates], q).flatten()
        top_idx = np.argpartition(dists, k)[:k]

        top_k = candidates[top_idx]
        top_dist = dists[top_idx]

        return [(k, dist) for k, dist in zip(top_k, top_dist)]

