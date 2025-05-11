from aida.aida import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import requests
import re
import ast

host = 'localhost'
dbname = 'benchbase'
user = 'test01'
passwd = 'password'
jobName = 'job1'
port = 55660


bench = "Torch Linear Regression"
subject = "Iterations over time (sequential)"

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

def getTimestamps(filename, offset_zero):
    with open(filename, 'r') as f:
        data = f.read()

    array = ast.literal_eval(data)

    if ( offset_zero == 0 ):
        offset_zero = array[0]
    offsets = [t - offset_zero for t in array]

    return offsets, offset_zero


job1 = "torchLinear1Seq"
job2 = "torchLinear2Seq"

y_gpu, offset_zero = getTimestamps(f"{job1}.txt", 0)
y_cpu, offset_zero = getTimestamps(f"{job2}.txt", offset_zero )

if len(y_cpu) != len(y_gpu):
    print(f"ARRAYS NOT EQUAL ({len(y_cpu)} vs {len(y_gpu)})")
    exit(0)

x = list(range(len(y_cpu))) 

#x = x[10:]  # Slice x to exclude the first 10 points
#y_cpu = y_cpu[10:]
#y_gpu = y_gpu[10:]

fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot latency time series
plt.plot(y_cpu,x, linestyle=':', linewidth=2, label=f"Job {job1}", color='blue')
plt.plot(y_gpu,x, linewidth=2, label=f"Job {job2}", color='green')

ymin, ymax = plt.ylim()
plt.vlines(x=[y_gpu[len(y_gpu)-1]], color='r', ymin=ymin, ymax=ymax, linestyle='dashed', label="GPU Switch")

plt.xlabel('Time (s)', fontsize=18, labelpad=20)  
plt.ylabel('Number of iterations', fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title(f"{subject} - {bench} Benchmark", fontsize=22)

ax1.grid()
plt.legend(fontsize=18)

plt.savefig(f"dbml/{bench}{subject}.pdf", format="pdf")
plt.show()
