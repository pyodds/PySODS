import taos
import sys
import random
import numpy
import numpy as np
import pandas as pd
import argparse
import time
import getpass
from utils.utils import output_performance,insert_data,connect_server,query_data,algorithm_selection


from algo.iForest import iForest

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Anomaly Detection Platform Settings")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--user', default='yli')
    parser.add_argument('--password', default='0906')
    parser.add_argument('--random_seed',default=42, type=int)
    parser.add_argument('--database',default='db')
    parser.add_argument('--table',default='t')
    parser.add_argument('--time_serie',default=False)
    parser.add_argument('--algorithm',default='robustcovariance',choices=['iforest','lof','ocsvm','robustcovariance'])
    args = parser.parse_args()

    #random seed setting
    rng = np.random.RandomState(args.random_seed)
    np.random.seed(args.random_seed)

    #password = getpass.getpass("Please input your password:")

    #connection configeration
    conn,cursor=connect_server(args.host, args.user, args.password)

    #read data
    print('Load dataset and table')
    start_time = time.clock()
    ground_truth=insert_data(conn,cursor,args.database,args.table)
    data = query_data(conn,cursor,args.database,args.table,args.time_serie)
    print('Loading cost: %.6f seconds' %(time.clock() - start_time))
    print('Load data successful')

    #algorithm
    clf = algorithm_selection(args.algorithm,random_state=rng)
    print('Start processing:')
    start_time = time.clock()
    clf.fit(data)
    prediction_result = clf.predict(data)

    output_performance(args.algorithm,ground_truth,prediction_result,time.clock() - start_time)


    conn.close()