import numpy as np

import sys
from

X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
clf = iForest(n_estimators=10, warm_start=True)
clf.fit(X)  # fit 10 trees
clf.set_params(n_estimators=20)  # add 10 more trees
clf.fit(X)  # fit the added trees