from multiprocessing import Pool 
import logging
import time
import numpy as np
import psutil

while True:

        # logging.info("start:{}:elapsed:{}".format(t1,np.mean(lengthArr)))
        with open('cpu_util.csv', 'a') as f:
            f.write(str(float(time.time())) +','+ str(psutil.cpu_percent(interval=0.2)) +'\n')

