from multiprocessing import Pool 
import logging
import time
import psycopg2
logging.basicConfig(level=logging.INFO, filename='query.log')
import numpy as np
import psutil
connection=psycopg2.connect(user='sf01',password='sf01',host='localhost',database='sf01')
cursor=connection.cursor()
print(cursor.execute("SELECT pg_backend_pid();"))
number = 10
lengthArr = []
cpuArr = []
index = 1

while True:
    cpu = float(psutil.cpu_percent())
    t1=time.time()
    cursor.execute("SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part WHERE p_partkey = l_partkey AND p_brand = 'Brand#13' AND p_container = 'JUMBO BAG' AND l_quantity < 10;")
    length = float(time.time() - t1)
    print(length)
    lengthArr.append(length)
    cpuArr.append(cpu)
    index  += 1
    if index == number:
        index = 0

        # logging.info("start:{}:elapsed:{}".format(t1,np.mean(lengthArr)))
        with open('result_gpu.csv', 'a') as f:
            f.write(str(t1)+','+str(np.mean(lengthArr)) +','+ str(np.mean(cpuArr))+'\n')
        lengthArr = []
        cpuArr = []


