import numpy as np
import numbers
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import datetime
import taos
import pandas as pd
from algo.iForest import iForest
from algo.ocsvm import ocsvm

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT

def output_performance(algorithm,ground_truth,y_pred):
    print ('='*30)
    print ('Results in Algorithm %s are:' %algorithm)
    print ('accuracy_score: %.2f' %accuracy_score(ground_truth, y_pred))
    print ('precision_score: %.2f' %precision_score(ground_truth, y_pred))
    print ('recall_score: %.2f' %recall_score(ground_truth, y_pred))
    print ('f1_score: %.2f' %f1_score(ground_truth, y_pred))
    print('=' * 30)

def insert_data(conn,consur,database,table):


    # Create a database named db
    try:
        consur.execute('drop database if exists %s' %database)
        consur.execute('create database if not exists %s' %database)
    except Exception as err:
        conn.close()
        raise (err)

    # use database
    try:
        consur.execute('use %s' %database)
    except Exception as err:
        conn.close()
        raise (err)

    # create table
    try:
        consur.execute('create table if not exists %s (ts timestamp, a float, b float)' %table)
    except Exception as err:
        conn.close()
        raise (err)

    start_time = datetime.datetime(2019, 8, 1)
    time_interval = datetime.timedelta(seconds=60)

    # insert data
    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.3 * np.random.randn(1)-2, 0.3 * np.random.randn(1)-2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table, start_time, 0.3 * np.random.randn(1)+2, 0.3 * np.random.randn(1)+2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(20):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time,np.random.uniform(low=-4, high=4), np.random.uniform(low=-4, high=4)))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    start_time = datetime.datetime(2019, 9, 1)
    time_interval = datetime.timedelta(seconds=60)

    # insert data
    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.1 * np.random.randn(1)-2, 0.1 * np.random.randn(1)-2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.1 * np.random.randn(1)+2, 0.1 * np.random.randn(1)+2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(20):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table, start_time,np.random.uniform(low=-4, high=4), np.random.uniform(low=-4, high=4)))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    n_outliers = 20
    ground_truth = np.ones(840, dtype=int)
    ground_truth[-n_outliers:] = -1
    ground_truth[400:420] = -1

    return ground_truth

def connect_server(host,user,password):
    # Connect to TDengine server.
    #
    # parameters:
    # @host     : TDengine server IP address
    # @user     : Username used to connect to TDengine server
    # @password : Password
    # @database : Database to use when connecting to TDengine server
    # @config   : Configuration directory
    conn = taos.connect(host,user,password,config="/etc/taos")
    cursor = conn.cursor()
    return conn,cursor

def query_data(conn,cursor,database,table,time_serie):

    # query data and return data in the form of list
    try:
        cursor.execute('select * from %s.%s' %(database,table))
    except Exception as err:
        conn.close()
        raise (err)

    # Column names are in c1.description list
    cols = cursor.description
    # Use fetchall to fetch data in a list
    data = cursor.fetchall()

    try:
        cursor.execute('select * from %s.%s' %(database,table))
    except Exception as err:
        conn.close()
        raise (err)

    a = pd.DataFrame(list(data))
    if time_serie:
        X = a
    else:
        X = a.iloc[:, 1:]
    return X

def output_performance(algorithm,ground_truth,y_pred):
    print ('='*30)
    print ('Results in Algorithm %s are:' %algorithm)
    print ('accuracy_score: %.2f' %accuracy_score(ground_truth, y_pred))
    print ('precision_score: %.2f' %precision_score(ground_truth, y_pred))
    print ('recall_score: %.2f' %recall_score(ground_truth, y_pred))
    print ('f1_score: %.2f' %f1_score(ground_truth, y_pred))
    print('=' * 30)

def insert_data(conn,consur,database,table):


    # Create a database named db
    try:
        consur.execute('drop database if exists %s' %database)
        consur.execute('create database if not exists %s' %database)
    except Exception as err:
        conn.close()
        raise (err)

    # use database
    try:
        consur.execute('use %s' %database)
    except Exception as err:
        conn.close()
        raise (err)

    # create table
    try:
        consur.execute('create table if not exists %s (ts timestamp, a float, b float)' %table)
    except Exception as err:
        conn.close()
        raise (err)

    start_time = datetime.datetime(2019, 8, 1)
    time_interval = datetime.timedelta(seconds=60)

    # insert data
    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.3 * np.random.randn(1)-2, 0.3 * np.random.randn(1)-2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table, start_time, 0.3 * np.random.randn(1)+2, 0.3 * np.random.randn(1)+2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(20):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time,np.random.uniform(low=-4, high=4), np.random.uniform(low=-4, high=4)))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    start_time = datetime.datetime(2019, 9, 1)
    time_interval = datetime.timedelta(seconds=60)

    # insert data
    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.1 * np.random.randn(1)-2, 0.1 * np.random.randn(1)-2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(200):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table,start_time, 0.1 * np.random.randn(1)+2, 0.1 * np.random.randn(1)+2))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    for i in range(20):
        try:
            consur.execute("insert into %s values ('%s', %f, %f,)" % (
            table, start_time,np.random.uniform(low=-4, high=4), np.random.uniform(low=-4, high=4)))
        except Exception as err:
            conn.close()
            raise (err)
        start_time += time_interval

    n_outliers = 20
    ground_truth = np.ones(840, dtype=int)
    ground_truth[-n_outliers:] = -1
    ground_truth[400:420] = -1

    return ground_truth

def connect_server(host,user,password):
    # Connect to TDengine server.
    #
    # parameters:
    # @host     : TDengine server IP address
    # @user     : Username used to connect to TDengine server
    # @password : Password
    # @database : Database to use when connecting to TDengine server
    # @config   : Configuration directory
    conn = taos.connect(host,user,password,config="/etc/taos")
    cursor = conn.cursor()
    return conn,cursor

def query_data(conn,cursor,database,table,time_serie):

    # query data and return data in the form of list
    try:
        cursor.execute('select * from %s.%s' %(database,table))
    except Exception as err:
        conn.close()
        raise (err)

    # Column names are in c1.description list
    cols = cursor.description
    # Use fetchall to fetch data in a list
    data = cursor.fetchall()

    try:
        cursor.execute('select * from %s.%s' %(database,table))
    except Exception as err:
        conn.close()
        raise (err)

    a = pd.DataFrame(list(data))
    if time_serie:
        X = a
    else:
        X = a.iloc[:, 1:]
    return X

def algorithm_selection(algorithm,random_state):
    algorithm_dic={'iforest':iForest(behaviour='new', max_samples='auto', random_state=random_state, contamination='auto'),'ocsvm':ocsvm(gamma='auto')}
    alg = algorithm_dic[algorithm]
    return alg