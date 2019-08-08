import numpy as np
import matplotlib.pyplot as plt


'''
example iforest
'''
# from algo.iForest import iForest
#
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
# clf = iForest(n_estimators=10, warm_start=True)
# clf.fit(X)  # fit 10 trees
# clf.set_params(n_estimators=20)  # add 10 more trees
# clf.fit(X)  # fit the added trees


'''
example lof
'''

# from algo.lof import LOF
# np.random.seed(42)
#
# # Generate train data
# X_inliers = 0.3 * np.random.randn(100, 2)
# X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
#
# # Generate some outliers
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# X = np.r_[X_inliers, X_outliers]
#
# n_outliers = len(X_outliers)
# ground_truth = np.ones(len(X), dtype=int)
# ground_truth[-n_outliers:] = -1
#
# # fit the model for outlier detection (default)
# clf = LOF(n_neighbors=20, contamination=0.1)
# # use fit_predict to compute the predicted labels of the training samples
# # (when LOF is used for outlier detection, the estimator has no predict,
# # decision_function and score_samples methods).
# y_pred = clf.fit_predict(X)
# n_errors = (y_pred != ground_truth).sum()
# X_scores = clf.negative_outlier_factor_
#
# plt.title("Local Outlier Factor (LOF)")
# plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# # plot circles with radius proportional to the outlier scores
# radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
# plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
#             facecolors='none', label='Outlier scores')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.xlabel("prediction errors: %d" % (n_errors))
# legend = plt.legend(loc='upper left')
# legend.legendHandles[0]._sizes = [10]
# legend.legendHandles[1]._sizes = [20]
# plt.show()
#
# print (X_scores)
# print(radius)

'''
example ocsvm
'''

# from algo.ocsvm import ocsvm
# # X = [[0], [0.44], [0.45], [0.46], [1]]
# # clf = ocsvm(gamma='auto',nu=0.2, kernel="rbf").fit(X)
# # y_pred=clf.predict(X)
# #
# # clf.score_samples(X)
# # print (y_pred)

'''
example robustcovariance
'''
# from algo.robustcovariance import robustcovariance
#
# X = [[0], [0.44], [0.45], [0.46], [1]]
# clf = robustcovariance(contamination=0.2).fit(X)
# y_pred=clf.predict(X)
#
# clf.score_samples(X)
# print (y_pred)
