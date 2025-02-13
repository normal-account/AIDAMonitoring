import pandas as pd
import pymonetdb
import sys
import pandas.io.sql as psql
import os
from time import time

config = __import__('TPCHconfig-pandas')
tpchqueries = __import__('TPCHqueries-pandas')

assert len(sys.argv) > 1, 'Usage: python3 runTPCH-pandas.py (<query_number>)...'
assert all(int(e) < 23 and int(e) >= 1 for e in sys.argv[1:]), 'Query numbers must be integers between 1 and 22'
queries = ['0' + str(int(e)) if int(e) < 10 else str(int(e)) for e in sys.argv[1:]]

class Database:

    def __init__(self, dbname, schema, host, user, pwd):
        self.con = pymonetdb.Connection(dbname, hostname=host, username=user, password=pwd, autocommit=True);
        self.schema = schema

    def setBufferSize(self, size):
        self.con.replysize = size

    def __getattr__(self, name):
        t = pd.DataFrame(psql.read_sql_query('SELECT * FROM {0}.{1};'.format(self.schema, name), self.con))
        self.__setattr__(name, t)
        return t

db = Database(config.dbname, config.schema, config.host, config.user, config.passwd)
db.setBufferSize(config.databuffersize)

os.system('mkdir -p {}'.format(config.outputDir))

if config.preload:
    print('----------[ Preloading ]----------')
    t0 = time()
    for t in ('lineitem', 'part', 'partsupp', 'customer', 'orders', 'nation', 'region', 'supplier'):
        getattr(db, t)
    t1 = time()
    print('Preloading all tables took {}s'.format(t1 - t0))

with open('{}/time-pandas.csv'.format(config.outputDir), 'a') as f:
    for q in queries:
        print('----------[ Query {0} ]----------'.format(q))
        t0 = time()
        r = getattr(tpchqueries, 'q' + q)(db)
        t1 = time()
        print(r)
        print('Executing query took {}s'.format(t1 - t0))
        f.write('{0},{1}\n'.format(int(q), t1 - t0))
