import sys
import os
from time import time
import pymonetdb

config = __import__('TPCHconfig-MonetDB')
tpchqueries = __import__('TPCHqueries-MonetDB')

assert len(sys.argv) > 1, 'Usage: python3 runTPCH-MonetDB.py (<query_number>)...'
assert all(int(e) < 23 and int(e) >= 1 for e in sys.argv[1:]), 'Query numbers must be integers between 1 and 22'
queries = ['0' + str(int(e)) if int(e) < 10 else str(int(e)) for e in sys.argv[1:]]

con = pymonetdb.connect(username=config.user, password=config.passwd, hostname=config.host, database=config.dbname)
cursor = con.cursor()
cursor.arraysize = config.databuffersize

os.system('mkdir -p {}'.format(config.outputDir))

with open('{}/time-MonetDB.csv'.format(config.outputDir), 'a') as f:
    for q in queries:
        print('----------[ Query {0} ]----------'.format(q))
        qlist = getattr(tpchqueries, 'q' + q)(config.schema)
        t0 = time()
        for query in qlist:
            cursor.execute(query)
        t1 = time()
        print(cursor.fetchall())
        print('Executing query took {}s'.format(t1 - t0))
        f.write('{0},{1}\n'.format(int(q), t1 - t0))
