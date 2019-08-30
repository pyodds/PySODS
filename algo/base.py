from sklearn.manifold import TSNE

class Base(object):
    def __init__(self):
        self.threshold=None
        pass


    def fit(self, X, y=None):
        """Fit detector. y is optional for unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        pass

    def predict(self, X):
        """Predict raw anomaly scores of X using the fitted detector.
        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        pass

    def decision_function(self,X):

        pass

