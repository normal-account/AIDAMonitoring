from aida.aida import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import requests
import re

host = 'localhost'
dbname = 'benchbase'
user = 'test01'
passwd = 'password'
jobName = 'job1'
port = 55660


bench = "Torch Linear Regression"
subject = "Latency"
key = 'Latency (microseconds)'

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

print("Waiting...")
hit = 0
while hit == 0:
    val = dw._getResponseTime()
    if None == val:
        hit = 0
    else:
        hit += 1
    time.sleep(0.5)

print("Starting measurement.")

i = 0
responseTime = -1

x = []
y_response = []
y_cpu = []
y_gpu = []

while responseTime != 0:
    i += 0.5

    # Getting response time
    responseTime = dw._getResponseTime()

    if responseTime == None:
        responseTime = 0

    y_response.append( responseTime )


    # Resetting pg_stat_statements
    dw._resetPgStatStatements()

    # Getting CPU usage
    cpu = dw._getCPUUsage()

    if cpu == None:
        cpu = 0

    y_cpu.append( cpu )


    # Getting GPU usage
    gpu = dw._getGPUUsage()

    if gpu == None:
        gpu = 0

    y_gpu.append( gpu )

    x.append( i )

    print( "Response time : " + str(responseTime) )

    time.sleep( 0.5 )


x = x[10:]  # Slice x to exclude the first 10 points
y_response = y_response[10:]
y_cpu = y_cpu[10:]
y_gpu = y_gpu[10:]

fig, ax1 = plt.subplots(figsize=(16, 9))

# Plot latency time series
plt.plot(x, y_response, linewidth=2, label=f"Response time (ms)", color='red')

plt.plot(x, y_cpu, linestyle=':', linewidth=2, label=f"CPU Usage (%)", color='blue')
plt.plot(x, y_gpu, linestyle=':', linewidth=2, label=f"GPU Usage (%)", color='green')

plt.xlabel('Time (s)', fontsize=18, labelpad=20)  
plt.ylabel('Latency (ms)', fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title(f"{subject} - {bench} Benchmark", fontsize=22)

ax1.grid()
plt.legend(fontsize=18)

#lt.savefig(f"plan/{bench}{subject}_test.pdf", format="pdf")
plt.show()