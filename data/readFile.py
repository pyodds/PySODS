import taos
import os
import os.path
import numpy as np
import pandas as pd
from string import digits
import argparse

from utils.utilities import connect_server

def read_File(Filename,db,tablename,cursor,connection,insert=False):
    # use database
    print(os.getcwd())
    if not os.path.exists(Filename):
        raise FileNotFoundError("%s not exist." %(Filename))
    try:
        cursor.execute('create database if not exists %s' %(db))
    except Exception as err:
        connection.close()
        raise (err)

    try:
        cursor.execute('use %s' %(db))
    except Exception as err:
        connection.close()
        raise (err)

    df=pd.read_csv(Filename)
    df_value=df.dtypes
    df_index=df_value.index

    table_list=[]
    for i in range(len(df_index)):
        table_list.append(df_index[i])
        table_list.append(df_value[i].name+', ')
    table_list[0]='ts'
    table_list[1]='timestamp, '
    table_list_str=' '.join(table_list)[:-2]
    remove_digits = str.maketrans('', '', digits)
    tabletitle = table_list_str.translate(remove_digits)

    try:
        cursor.execute('create table if not exists %s (%s)' %(tablename,tabletitle))
    except Exception as err:
        connection.close()
        raise (err)

    if insert:
        try:
            cursor.execute('insert into  %s.%s file %s' % (db,tablename, Filename))
        except Exception as err:
            connection.close()
            raise (err)

    elif not insert:
        try:
            cursor.execute('import into %s.%s file %s' % (db,tablename, Filename))
        except Exception as err:
            connection.close()
            raise (err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reading CSV to Tables")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--user', default='yli')
    parser.add_argument('--password', default='0906')
    parser.add_argument('--random_seed',default=42, type=int)
    parser.add_argument('--database',default='db')
    parser.add_argument('--table',default='tt')
    parser.add_argument('--file_name',default='demo2.csv')
    parser.add_argument('--insert',default=False)

    args = parser.parse_args()
    connection,cursor=connect_server(args.host, args.user, args.password)

    read_File(args.file_name, args.database, args.table, cursor, connection, insert=False)

    connection.close()
