import numpy as np
from algo.base import Base
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
from numpy import percentile


class CBLOF(Base):

    def __init__(self, n_clusters=8, contamination=0.1,
                 clustering_estimator=None, alpha=0.9, beta=5,
                 use_weights=False, random_state=None,
                 n_jobs=1):
        super(CBLOF, self).__init__()
        self.n_clusters = n_clusters
        self.clustering_estimator = clustering_estimator
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.contamination=contamination
    # noinspection PyIncorrectDocstring
    def fit(self, X,y=None):
        """Fit detector. y is optional for unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        X=X.to_numpy()
        # validate inputs X and y (optional)
        n_samples, n_features = X.shape

        # check parameters
        # number of clusters are default to 8
        self._validate_estimator(default=KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_jobs=self.n_jobs))

        self.clustering_estimator_.fit(X=X, y=y)
        # Get the labels of the clustering results
        # labels_ is consistent across sklearn clustering algorithms
        self.cluster_labels_ = self.clustering_estimator_.labels_
        self.cluster_sizes_ = np.bincount(self.cluster_labels_)

        # Get the actual number of clusters
        self.n_clusters_ = self.cluster_sizes_.shape[0]

        if self.n_clusters_ != self.n_clusters:
            warnings.warn("The chosen clustering for CBLOF forms {0} clusters"
                          "which is inconsistent with n_clusters ({1}).".
                          format(self.n_clusters_, self.n_clusters))

        self._set_cluster_centers(X, n_features)
        self._set_small_large_clusters(n_samples)

        self.decision_scores_ = self._decision_function(X,
                                                        self.cluster_labels_)

        self._process_decision_scores()
        return

    def _validate_estimator(self, default=None):
        """Check the value of alpha and beta and clustering algorithm.
        """

        if self.clustering_estimator is not None:
            self.clustering_estimator_ = self.clustering_estimator
        else:
            self.clustering_estimator_ = default

        # make sure the base clustering algorithm is valid
        if self.clustering_estimator_ is None:
            raise ValueError("clustering algorithm cannot be None")

    def _process_decision_scores(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - labels_: binary labels of training data
        Returns
        -------
        self
        """

        self.threshold_ = percentile(self.decision_scores_,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def predict(self, X):
        X=X.to_numpy()
        anomalies = self.decision_function(X)
        ranking = np.sort(anomalies)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        mask = (anomalies>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        labels = self.clustering_estimator_.predict(X)
        return self._decision_function(X, labels)
    def _set_cluster_centers(self, X, n_features):
        # Noted not all clustering algorithms have cluster_centers_
        if hasattr(self.clustering_estimator_, 'cluster_centers_'):
            self.cluster_centers_ = self.clustering_estimator_.cluster_centers_
        else:
            # Set the cluster center as the mean of all the samples within
            # the cluster
            warnings.warn("The chosen clustering for CBLOF does not have"
                          "the center of clusters. Calculate the center"
                          "as the mean of the clusters.")
            self.cluster_centers_ = np.zeros([self.n_clusters_, n_features])
            for i in range(self.n_clusters_):
                self.cluster_centers_[i, :] = np.mean(
                    X[np.where(self.cluster_labels_ == i)], axis=0)

    def _set_small_large_clusters(self, n_samples):
        # Sort the index of clusters by the number of samples belonging to it
        size_clusters = np.bincount(self.cluster_labels_)

        # Sort the order from the largest to the smallest
        sorted_cluster_indices = np.argsort(size_clusters * -1)

        # Initialize the lists of index that fulfill the requirements by
        # either alpha or beta
        alpha_list = []
        beta_list = []

        for i in range(1, self.n_clusters_):
            temp_sum = np.sum(size_clusters[sorted_cluster_indices[:i]])
            if temp_sum >= n_samples * self.alpha:
                alpha_list.append(i)

            if size_clusters[sorted_cluster_indices[i - 1]] / size_clusters[
                sorted_cluster_indices[i]] >= self.beta:
                beta_list.append(i)

            # Find the separation index fulfills both alpha and beta
        intersection = np.intersect1d(alpha_list, beta_list)

        if len(intersection) > 0:
            self._clustering_threshold = intersection[0]
        elif len(alpha_list) > 0:
            self._clustering_threshold = alpha_list[0]
        elif len(beta_list) > 0:
            self._clustering_threshold = beta_list[0]
        else:
            raise ValueError("Could not form valid cluster separation. Please "
                             "change n_clusters or change clustering method")

        self.small_cluster_labels_ = sorted_cluster_indices[
                                     self._clustering_threshold:]
        self.large_cluster_labels_ = sorted_cluster_indices[
                                     0:self._clustering_threshold]

        # No need to calculate small cluster center
        # self.small_cluster_centers_ = self.cluster_centers_[
        #     self.small_cluster_labels_]

        self._large_cluster_centers = self.cluster_centers_[
            self.large_cluster_labels_]

    def _decision_function(self, X, labels):
        # Initialize the score array
        scores = np.zeros([X.shape[0], ])

        small_indices = np.where(
            np.isin(labels, self.small_cluster_labels_))[0]
        large_indices = np.where(
            np.isin(labels, self.large_cluster_labels_))[0]

        if small_indices.shape[0] != 0:
            # Calculate the outlier factor for the samples in small clusters
            dist_to_large_center = cdist(X[small_indices, :],
                                         self._large_cluster_centers)

            scores[small_indices] = np.min(dist_to_large_center, axis=1)

        if large_indices.shape[0] != 0:
            # Calculate the outlier factor for the samples in large clusters
            large_centers = self.cluster_centers_[labels[large_indices]]

            scores[large_indices] = pairwise_distances_no_broadcast(
                X[large_indices, :], large_centers)

        if self.use_weights:
            # Weights are calculated as the number of elements in the cluster
            scores = scores * self.cluster_sizes_[labels]

        return scores.ravel()

def pairwise_distances_no_broadcast(X, Y):
    """Utility function to calculate row-wise euclidean distance of two matrix.
    Different from pair-wise calculation, this function would not broadcast.
    For instance, X and Y are both (4,3) matrices, the function would return
    a distance vector with shape (4,), instead of (4,4).
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Row-wise euclidean distance of X and Y
    """

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)


def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    """Internal function for calculating the distance with numba. Do not use.
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples
    Y : array of shape (n_samples, n_features)
        Second input samples
    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.
    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

