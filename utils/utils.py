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
from algo.staticautoencoder import StaticAutoEncoder
from algo.Luminol import LuminolDet
from algo.cblof import CBLOF
from algo.knn import KNN
from algo.hbos import HBOS
from algo.sod import SOD
from algo.pca import PCA
from sklearn.metrics import roc_auc_score
from algo.dagmm import DAGMM
from algo.lstm_ad import LSTMAD
from algo.lstm_enc_dec_axl import LSTMED
from algo.autoencoder import AutoEncoder


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

    X.fillna(method='ffill')
    X.fillna(method='bfill')


    return X,new_ground_truth

def algorithm_selection(algorithm,random_state,contamination):
    algorithm_dic={'iforest':iForest(contamination=contamination,n_estimators=100,max_samples="auto", max_features=1.,bootstrap=False,n_jobs=None,behaviour='old',random_state=random_state,verbose=0,warm_start=False),
                   'ocsvm':ocsvm(gamma='auto',kernel='rbf', degree=3,coef0=0.0, tol=1e-3, nu=0.5, shrinking=True, cache_size=200,verbose=False, max_iter=-1, random_state=random_state),
                   'lof': LOF(contamination=contamination,n_neighbors=20, algorithm='auto', leaf_size=30,metric='minkowski', p=2, metric_params=None, novelty=True, n_jobs=None),
                   'robustcovariance':robustcovariance(random_state=random_state,store_precision=True, assume_centered=False,support_fraction=None, contamination=0.1),
                   'staticautoencoder':StaticAutoEncoder(contamination=contamination,epoch=100,dropout_rate=0.2,regularizer_weight=0.1,activation='relu',kernel_regularizer=0.01,loss_function='mse',optimizer='adam'),
                   'cblof':CBLOF(contamination=contamination,n_clusters=8, clustering_estimator=None, alpha=0.9, beta=5,use_weights=False, random_state=random_state,n_jobs=1),
                   'knn':KNN(contamination=contamination,n_neighbors=5, method='largest',radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1),
                   'hbos':HBOS(contamination=contamination, n_bins=10, alpha=0.1, tol=0.5),
                   'sod':SOD(contamination=contamination,n_neighbors=20, ref_set=10,alpha=0.8),
                   'pca':PCA(contamination=contamination, n_components=None, n_selected_components=None, copy=True, whiten=False, svd_solver='auto',tol=0.0, iterated_power='auto',random_state=random_state,weighted=True, standardization=True),
                   'dagmm':DAGMM(contamination=contamination,num_epochs=10, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-3, batch_size=50, gmm_k=3, normal_percentile=80, sequence_length=30, autoencoder_args=None),
                   'luminol': LuminolDet(contamination=contamination),
                   'autoencoder':AutoEncoder(contamination=contamination,num_epochs=10, batch_size=20, lr=1e-3,hidden_size=5, sequence_length=30, train_gaussian_percentage=0.25),
                   'lstm_ad':LSTMAD(contamination=contamination,len_in=1, len_out=10, num_epochs=100, lr=1e-3, batch_size=1),
                   'lstm_ed':LSTMED(contamination=contamination,num_epochs=10, batch_size=20, lr=1e-3,hidden_size=5, sequence_length=30, train_gaussian_percentage=0.25)
                   }
    alg = algorithm_dic[algorithm]
    return alg



