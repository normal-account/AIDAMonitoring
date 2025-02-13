import math;
import numpy as np;

import pymonetdb.sql;

import pandas.io.sql as psql;
import pandas;

from aida.aida import *;



import os;

host='localhost'
dbname='test01'
user='test01'
passwd='test01'
jobName=None
port=55660

#number=1;
repeat=1;
warmup=0;

arraysize = 1000000;

def thisJobName(filename):
    return os.path.basename(filename);

def getDBC(jobName):
    return AIDA.connect(host, dbname, user, passwd, jobName, port);

def getNumPyOpt(con, tableName):
    cursor = con.cursor();
    cursor.arraysize = arraysize;

    rows = cursor.execute('select * from {};'.format(tableName));
    data = cursor.fetchall();

    arr =  np.asarray(data);
    return arr;


def getNumPy(con, tableName):
    cursor = con.cursor();
    cursor.arraysize = 100;

    rows = cursor.execute('select * from {};'.format(tableName));

    data = [];
    for i in range(0, math.ceil(rows/cursor.arraysize)):
        data += cursor.fetchmany();

    arr =  np.asarray(data);
    return arr;

def getTabularDataMatrix(dbc, tableName):
    tbl = dbc._getDBTable(tableName);
    tbl.loadData(matrix=True);
    return tbl;


def getTabularData(dbc, tableName):
    tbl = dbc._getDBTable(tableName);
    tbl.loadData();
    return tbl;

def getPandasDF(con, tableName):
    con.replysize = 100;
    sql = 'select * from {};'.format(tableName);
    df = psql.read_sql_query(sql=sql, con=con);
    return df;

def getPandasDFOpt(con, tableName):
    con.replysize = arraysize;
    sql = 'select * from {};'.format(tableName);
    df = psql.read_sql_query(sql=sql, con=con);
    return df;