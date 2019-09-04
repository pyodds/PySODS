import luminol
from .base import Base
from luminol import anomaly_detector
import numpy as np
import pandas as pd
from luminol.modules.time_series import TimeSeries
from luminol.utils import to_epoch
import datetime
from sklearn.decomposition import IncrementalPCA

class luminolDet(Base):
    def __init__(self,contamination=0.1):
        self.contamination=contamination

    def fit(self,ts):
        # a=str(ts[:,0])
        ts=ts.to_numpy()
        timestamp = np.asarray(ts[:,0].astype(np.datetime64))
        pca = IncrementalPCA(n_components=1)
        value=np.reshape(pca.fit_transform(ts[:,1:]),-1)
        ts = pd.Series(value, index=timestamp)
        ts.index = ts.index.map(lambda d: to_epoch(str(d)))
        lts = TimeSeries(ts.to_dict())
        self.ts=timestamp
        self.ts_value=value
        self.detector = anomaly_detector.AnomalyDetector(lts)

        return self

    def predict(self,ts):
        anomalies = np.reshape(self.detector.get_all_scores().values,-1)
        self.decision=anomalies
        ranking = np.sort(anomalies)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        self.threshold = threshold
        mask = (anomalies>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

    def decision_function(self,ts):
        return self.decision




