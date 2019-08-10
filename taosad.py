import taos
import sys
import datetime
import random
import numpy
import numpy as np
import matplotlib.pyplot as plt


from algo.iForest import iForest
rng = np.random.RandomState(42)

start_time = datetime.datetime(2019, 7, 1)
time_interval = datetime.timedelta(seconds=60)

# Connect to TDengine server.
#
# parameters:
# @host     : TDengine server IP address
# @user     : Username used to connect to TDengine server
# @password : Password
# @database : Database to use when connecting to TDengine server
# @config   : Configuration directory
conn = taos.connect(host="127.0.0.1", user="yli", password="0906")
c1 = conn.cursor()
c1.execute('create database trane')
c1.execute('use trane')

c1.execute('create table if not exists tb1 ( a float, b float)')
c1.execute('insert into tb1 file test1.csv')
c1.execute('select * from trane.tb1')

X_train  = c1.fetchall()

X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = iForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
