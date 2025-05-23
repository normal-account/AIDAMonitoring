import sys
import os
from time import time

config = __import__('TPCHconfig-AIDA')
tpchqueries = __import__('TPCHqueries-AIDA')

assert len(sys.argv) > 1, 'Usage: python3 runTPCH-AIDA.py (<query_number>)...'
assert all(int(e) < 23 and int(e) >= 1 for e in sys.argv[1:]), 'Query numbers must be integers between 1 and 22'
queries = ['0' + str(int(e)) if int(e) < 10 else str(int(e)) for e in sys.argv[1:]]

db = config.getDBC(config.jobName);

os.system('mkdir -p {}'.format(config.outputDir))

cols =  {'region': ('r_regionkey', 'r_name', 'r_comment'),
         'nation': ('n_nationkey', 'n_name', 'n_regionkey', 'n_comment'),
         'part': ('p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'),
         'supplier': ('s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'),
         'partsupp': ('ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'),
         'customer': ('c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'),
         'orders': ('o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'),
         'lineitem': ('l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'),
        }

class Database:

    def __init__(self, db):
        self.db = db

    def __getattr__(self, name):
        t = getattr(self.db, name)
        try:
            t = t.project(cols[name]) * 1
            self.__setattr__(name, t)
        except KeyError:
            self.__setattr__(name, t)
        return t

if config.udfVSvtable:
    print('----------[ Preloading ]----------')
    t0 = time()
    db2 = Database(db)
    for t in ('lineitem', 'part', 'partsupp', 'customer', 'orders', 'nation', 'region', 'supplier'):
        t = getattr(db, t)
        t.loadData()
    t1 = time()
    print('Preloading all tables took {}s'.format(t1 - t0))
else:
    db2 = db

with open('{}/time-AIDA.csv'.format(config.outputDir), 'a') as f:
    for q in queries:
        print('----------[ Query {0} ]----------'.format(q))
        t0 = time()
        r = getattr(tpchqueries, 'q' + q)(db2)
        if(hasattr(r, '_genSQL_')):
            r.loadData()
        t1 = time()
        if(hasattr(r, '_genSQL_')):
            print(r.rows)
        else:
            print(r)
        print('Executing query took {}s'.format(t1 - t0))
        f.write('{0},{1}\n'.format(int(q), t1 - t0))
