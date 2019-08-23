import taos
import sys
import datetime
import random
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



from algo.iForest import iForest
rng = np.random.RandomState(42)

start_time = datetime.datetime(2019, 8, 1)
time_interval = datetime.timedelta(seconds=60)

np.random.seed(42)

# Connect to TDengine server.
#
# parameters:
# @host     : TDengine server IP address
# @user     : Username used to connect to TDengine server
# @password : Password
# @database : Database to use when connecting to TDengine server
# @config   : Configuration directory
conn = taos.connect(host="127.0.0.1", user="yli", password="0906",config="/etc/taos")
# c1 = conn.cursor()
# #c1.execute('create database trane')
# c1.execute('use trane')
#
# c1.execute('create table if not exists tb1 (ts timestamp, a float, b float, c binary(20))')
# #c1.execute('insert into tb1 file test1.csv')
# c1.execute('select * from trane.tb1')
#
# X_train  = c1.fetchall()
# cols = c1.description
#
# for i in c1:
#     print (i)
# print (X_train)
# conn.close()
#
# X = 0.3 * rng.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
#
# X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
#
# # fit the model
# clf = iForest(behaviour='new', max_samples=100,
#                       random_state=rng, contamination='auto')
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
#
# # plot the line, the samples, and the nearest vectors to the plane
# xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.title("IsolationForest")
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
#                  s=20, edgecolor='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
#                  s=20, edgecolor='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
#                 s=20, edgecolor='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([b1, b2, c],
#            ["training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left")
# plt.show()

c1 = conn.cursor()

# Create a database named db
try:
    c1.execute('drop database if exists db')
    c1.execute('create database if not exists db')
except Exception as err:
    conn.close()
    raise (err)

# use database
try:
    c1.execute('use db')
except Exception as err:
    conn.close()
    raise (err)

# create table
try:
    c1.execute('create table if not exists t (ts timestamp, a float, b float)')
except Exception as err:
    conn.close()
    raise (err)

# insert data
for i in range(200):
    try:
        c1.execute("insert into t values ('%s', %f, %f,)" % (
        start_time, 0.3 * np.random.randn(1)-2, 0.3 * np.random.randn(1)-2))
    except Exception as err:
        conn.close()
        raise (err)
    start_time += time_interval

for i in range(200):
    try:
        c1.execute("insert into t values ('%s', %f, %f,)" % (
        start_time, 0.3 * np.random.randn(1)+2, 0.3 * np.random.randn(1)+2))
    except Exception as err:
        conn.close()
        raise (err)
    start_time += time_interval

for i in range(20):
    try:
        c1.execute("insert into t values ('%s', %f, %f,)" % (
        start_time,np.random.uniform(low=-4, high=4), np.random.uniform(low=-4, high=4)))
    except Exception as err:
        conn.close()
        raise (err)
    start_time += time_interval


# query data and return data in the form of list
try:
    c1.execute('select * from db.t')
except Exception as err:
    conn.close()
    raise (err)

# Column names are in c1.description list
cols = c1.description
# Use fetchall to fetch data in a list
data = c1.fetchall()

try:
    c1.execute('select * from db.t')
except Exception as err:
    conn.close()
    raise (err)

# #Use iterator to go through the retreived data
# for col in c1:
#     print(col)
#



a = pd.DataFrame(list(data))
X = a.iloc[:,1:]




clf = iForest(behaviour='new', max_samples='auto',
                      random_state=rng, contamination='auto')
clf.fit(X)
y_pred_train = clf.predict(X)



n_outliers = 20
ground_truth = np.ones(420, dtype=int)
ground_truth[-n_outliers:] = -1

print (accuracy_score(ground_truth, y_pred_train))
c1.execute('DROP TABLE t')

conn.close()
def exitProgram(conn):
    conn.close()
    sys.exit()