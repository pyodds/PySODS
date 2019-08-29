import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from numpy import percentile

from .base import Base
from utils.utilities import check_parameter

class SOD(Base):
    """Subspace outlier detection (SOD) schema aims to detect outlier in
    varying subspaces of a high dimensional feature space. For each data
    object, SOD explores the axis-parallel subspace spanned by the data
    object's neighbors and determines how much the object deviates from the
    neighbors in this subspace.
    See :cite:`kriegel2009outlier` for details.
    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for k neighbors queries.
    ref_set: int, optional (default=10)
        specifies the number of shared nearest neighbors to create the
        reference set. Note that ref_set must be smaller than n_neighbors.
    alpha: float in (0., 1.), optional (default=0.8)
           specifies the lower limit for selecting subspace.
           0.8 is set as default as suggested in the original paper.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, contamination=0.1, n_neighbors=20, ref_set=10,
                 alpha=0.8):
        super(SOD, self).__init__()
        if isinstance(n_neighbors, int):
            check_parameter(n_neighbors, low=1, param_name='n_neighbors')
        else:
            raise ValueError(
                "n_neighbors should be int. Got %s" % type(n_neighbors))

        if isinstance(ref_set, int):
            check_parameter(ref_set, low=1, high=n_neighbors,
                            param_name='ref_set')
        else:
            raise ValueError("ref_set should be int. Got %s" % type(ref_set))

        if isinstance(alpha, float):
            check_parameter(alpha, low=0.0, high=1.0, param_name='alpha')
        else:
            raise ValueError("alpha should be float. Got %s" % type(alpha))

        self.n_neighbors_ = n_neighbors
        self.ref_set_ = ref_set
        self.alpha_ = alpha
        self.decision_scores_ = None
        self.contamination=contamination

    def fit(self, X, y=None):
        """Fit detector. y is optional for unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """

        # validate inputs X and y (optional)
        X = X.to_numpy()
        X = check_array(X)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return self

    def predict(self, X):
        X = X.to_numpy()
        anomalies =self.decision_function(X)
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
        return self._sod(X)

    def _snn(self, X):
        """This function is called internally to calculate the shared nearest
        neighbors (SNN). SNN is reported to be more robust than k nearest
        neighbors.
        Returns
        -------
        snn_indices : numpy array of shape (n_shared_nearest_neighbors,)
            The indices of top k shared nearest neighbors for each observation.
        """
        knn = NearestNeighbors(n_neighbors=self.n_neighbors_)
        knn.fit(X)
        # Get the knn index
        ind = knn.kneighbors(return_distance=False)
        _count = np.zeros(shape=(ind.shape[0], self.ref_set_), dtype=np.uint16)
        # Count the distance
        for i in range(ind.shape[0]):
            temp = np.sum(np.isin(ind, ind[i]), axis=1).ravel()
            temp[i] = np.iinfo(np.uint16).max
            _count[i] = np.argsort(temp)[::-1][1:self.ref_set_ + 1]

        return _count

    def _sod(self, X):
        """This function is called internally to perform subspace outlier
        detection algorithm.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        ref_inds = self._snn(X)
        anomaly_scores = np.zeros(shape=(X.shape[0],))

        anomaly_scores = np.zeros(shape=(X.shape[0],))
        for i in range(X.shape[0]):
            obs = X[i]
            ref = X[ref_inds[i,],]
            means = np.mean(ref, axis=0)  # mean of each column
            # average squared distance of the reference to the mean
            var_total = np.sum(np.sum(np.square(ref - means))) / self.ref_set_
            var_expect = self.alpha_ * var_total / X.shape[1]
            var_actual = np.var(ref, axis=0)  # variance of each attribute
            var_inds = [1 if (j < var_expect) else 0 for j in var_actual]
            rel_dim = np.sum(var_inds)
            if rel_dim != 0:
                anomaly_scores[i] = np.sqrt(
                    np.dot(var_inds, np.square(obs - means)) / rel_dim)

        return anomaly_scores
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


