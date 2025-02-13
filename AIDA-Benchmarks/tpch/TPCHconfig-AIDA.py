import os
from aida.aida import *

host = 'localhost'
dbname = 'tpchsf01'
user = 'tpchsf01'
passwd = 'tpchsf01'
jobName = 'tpch'
port = 55660

SF = 0.01 #used by query 11. indicate the scale factor of the tpch database.

udfVSvtable = True

outputDir = 'output'

def thisJobName(filename):
    return os.path.basename(filename);

def getDBC(jobName):
    return AIDA.connect(host, dbname, user, passwd, jobName, port);
