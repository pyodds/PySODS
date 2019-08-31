import numpy as np
import numbers
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import datetime
import taos
import pandas as pd
from algo.iForest import iForest
from algo.ocsvm import ocsvm
from algo.lof import LOF
from algo.robustcovariance import robustcovariance
from algo.AutoEncoder import AutoEncoder
from algo.Luminol import LuminolDetec
from algo.cblof import CBLOF
from algo.knn import KNN
from algo.hbos import HBOS
from algo.sod import SOD
from algo.pca import PCA
from sklearn.metrics import roc_auc_score


def insert_demo_data(conn,consur,database,table,start_time,end_time,time_serie):


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



def output_performance(algorithm,ground_truth,y_pred,time,outlierness):
    print ('='*30)
    print ('Results in Algorithm %s are:' %algorithm)
    print ('accuracy_score: %.2f' %accuracy_score(ground_truth, y_pred))
    print ('precision_score: %.2f' %precision_score(ground_truth, y_pred))
    print ('recall_score: %.2f' %recall_score(ground_truth, y_pred))
    print ('f1_score: %.2f' %f1_score(ground_truth, y_pred))
    print ('processing time: %.6f seconds' %time)
    print ('roc_auc_score: %.2f' %max(roc_auc_score(ground_truth, outlierness),1-roc_auc_score(ground_truth, outlierness)))
    print('=' * 30)

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

def query_data(conn,cursor,database,table,time_serie,start_time,end_time,ground_truth,time_serie_name):

    # query data and return data in the form of list
    if start_time and end_time:
        try:
            cursor.execute("select * from %s.%s where %s >= \'%s\' and %s <= \'%s\' " %(database,table,time_serie_name,start_time,time_serie_name,end_time))
        except Exception as err:
            conn.close()
            raise (err)
    elif not start_time and not end_time:
        try:
            cursor.execute('select * from %s.%s' %(database,table))
        except Exception as err:
            conn.close()
            raise (err)
    elif start_time and not end_time:
        try:
            cursor.execute("select * from %s.%s where %s >=  \'%s\' " %(database,table,time_serie_name,start_time))
        except Exception as err:
            conn.close()
            raise (err)
    elif not start_time and  end_time:
        try:
            cursor.execute("select * from %s.%s where %s <=  \'%s\' " %(database,table,time_serie_name,end_time))
        except Exception as err:
            conn.close()
            raise (err)

    # Column names are in c1.description list
    cols = cursor.description
    # Use fetchall to fetch data in a list
    data = cursor.fetchall()

    if start_time and end_time:
        try:
            cursor.execute("select * from %s.%s where %s >=  \'%s\' and %s <=  \'%s\' " %(database,table,time_serie_name,start_time,time_serie_name,end_time))
        except Exception as err:
            conn.close()
            raise (err)
    elif not start_time and not end_time:
        try:
            cursor.execute('select * from %s.%s' %(database,table))
        except Exception as err:
            conn.close()
            raise (err)
    elif start_time and not end_time:
        try:
            cursor.execute("select * from %s.%s where %s >=  \'%s\' " %(database,table,time_serie_name,start_time))
        except Exception as err:
            conn.close()
            raise (err)
    elif not start_time and  end_time:
        try:
            cursor.execute("select * from %s.%s where %s <=  \'%s\' " %(database,table,time_serie_name,end_time))
        except Exception as err:
            conn.close()
            raise (err)

    tmp = pd.DataFrame(list(data))

    if time_serie:
        X = tmp
    else:
        X = tmp.iloc[:, 1:]

    if True:
        try:
            cursor.execute('select * from %s.%s' %(database,table))
        except Exception as err:
            conn.close()
            raise (err)
        whole_data = cursor.fetchall()
        try:
            cursor.execute('select * from %s.%s' %(database,table))
        except Exception as err:
            conn.close()
            raise (err)

        whole_tmp = pd.DataFrame(list(whole_data))

        # ground_truth_mask= data[:,0]>=args.start_time and np.where(data[:,0]<=args.endtime
        # ground_truth2=ground_truth[ground_truth_mask]
        timestamp=np.array(whole_tmp.ix[:,0].to_numpy(), dtype='datetime64')
        timestamp=np.reshape(timestamp,-1)
        new_ground_truth=[]
        if start_time and end_time:
            for i in range(len(whole_tmp)):
                if timestamp[i]>=np.datetime64(start_time) and timestamp[i]<=np.datetime64(end_time):
                    new_ground_truth.append(ground_truth[i])
        elif start_time and not end_time:
            for i in range(len(whole_tmp)):
                if timestamp[i]>=np.datetime64(start_time):
                    new_ground_truth.append(ground_truth[i])
        elif not start_time and  end_time:
            for i in range(len(whole_tmp)):
                if timestamp[i]<=np.datetime64(end_time):
                    new_ground_truth.append(ground_truth[i])
        elif not start_time and not end_time:
            new_ground_truth=ground_truth
        new_ground_truth=np.array(new_ground_truth)
    else:
        new_ground_truth=ground_truth

    return X,new_ground_truth

def algorithm_selection(algorithm,random_state,contamination):
    algorithm_dic={'iforest':iForest(behaviour='new', max_samples='auto', random_state=random_state, contamination='auto'),
                   'ocsvm':ocsvm(gamma='auto'),
                   'lof': LOF(contamination=contamination,novelty=True),
                   'robustcovariance':robustcovariance(random_state=random_state),
                   'robustautoencoder':AutoEncoder(contamination=contamination),
                   'luminol':LuminolDetec(contamination=contamination),
                   'cblof':CBLOF(contamination=contamination),
                   'knn':KNN(contamination=contamination),
                   'hbos':HBOS(contamination=contamination),
                   'sod':SOD(contamination=contamination),
                   'pca':PCA(contamination=contamination)}
    alg = algorithm_dic[algorithm]
    return alg



