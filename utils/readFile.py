import taos
import os.path
import numpy as np
import pandas as pd
def read_File(Filename,db,tablename,cursor,connection,insert=False):
    # use database
    if not os.path.exists(Filename):
        raise FileNotFoundError("%s not exist." % (Filename))
    try:
        cursor.execute('use db')
    except Exception as err:
        connection.close()
        raise (err)

    # create table
    data = pd.read_csv("filename")

    try:
        cursor.execute('create table if not exists tablename (ts timestamp, a float, b float)')
    except Exception as err:
        connection.close()
        raise (err)

    if not insert:
        # load table
        cursor.execute('import into tablename file %s' %(Filename))
    else:
        cursor.execute('insert into tablename file %s' %(Filename))



