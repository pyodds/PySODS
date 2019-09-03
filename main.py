import taos
import sys
import random
import numpy
import numpy as np
import pandas as pd
import argparse
import time
import logging
import getpass
from utils.utilities import output_performance,insert_demo_data,connect_server,query_data
from utils.import_algorithm import algorithm_selection
from utils.plot_utils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
logging.disable(logging.WARNING)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Anomaly Detection Platform Settings")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--user', default='yli')
    parser.add_argument('--password', default='0906')
    parser.add_argument('--random_seed',default=42, type=int)
    parser.add_argument('--database',default='db')
    parser.add_argument('--table',default='t')
    parser.add_argument('--time_serie',default=False)
    parser.add_argument('--visualize_distribution',default=True)
    parser.add_argument('--algorithm',default='luminol',choices=['iforest','lof','ocsvm','robustcovariance','staticautoencoder','luminol','cblof','knn','hbos','sod','pca','dagmm','autoencoder','lstm_ad','lstm_ed'])
    parser.add_argument('--contamination',default=0.05)
    parser.add_argument('--start_time',default='2019-07-20 00:00:00')
    parser.add_argument('--end_time',default='2019-08-20 00:00:00')
    parser.add_argument('--time_serie_name',default='ts')



    args = parser.parse_args()

    #random seed setting
    rng = np.random.RandomState(args.random_seed)
    np.random.seed(args.random_seed)

    # args.password = getpass.getpass("Please input your password:")

    #connection configeration
    conn,cursor=connect_server(args.host, args.user, args.password)

    #read data
    print('Load dataset and table')
    start_time = time.clock()
    ground_truth_whole=insert_demo_data(conn,cursor,args.database,args.table,args.start_time,args.end_time,args.time_serie)

    data,ground_truth = query_data(conn,cursor,args.database,args.table,
                                   args.time_serie,args.start_time,args.end_time,ground_truth_whole,args.time_serie_name)

    print('Loading cost: %.6f seconds' %(time.clock() - start_time))
    print('Load data successful')

    #algorithm

    clf = algorithm_selection(args.algorithm,random_state=rng,contamination=args.contamination)
    print('Start processing:')
    start_time = time.clock()
    clf.fit(data)
    prediction_result = clf.predict(data)
    outlierness = clf.decision_function(data)

    output_performance(args.algorithm,ground_truth,prediction_result,time.clock() - start_time,outlierness)

    if args.visualize_distribution:
        if not args.time_serie:
            visualize_distribution_static(data,prediction_result,outlierness)
            visualize_distribution(data,prediction_result,outlierness)
            visualize_outlierscore(outlierness,prediction_result,args.contamination)
        else:
            visualize_distribution_time_serie(clf.ts,data)
            visualize_outlierscore(outlierness,prediction_result,args.contamination)


    conn.close()