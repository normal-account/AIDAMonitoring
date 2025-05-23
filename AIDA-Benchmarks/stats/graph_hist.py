import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

def genCSV(folder, filename):
    path = folder + "/" + filename
    # Open the output file generated by perf sched timehist
    with open(f"{path}.txt", 'r') as f:
        data = f.readlines()

    new_path = f"hist/{filename}.csv"
    with open(new_path, 'w+') as outfile:
        outfile.write("time,cpu,task,wait_time,sch_delay,run_time\n")
        line_num = 0
        for line in data:
            if line_num < 3:
                line_num += 1
                continue

            match = re.match(r"\s+(\d+\.\d+)\s+\[(\d+)\]\s+postgres\[(\d+)\]\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", line)
            if match:
                outfile.write(f"{match.group(1)},{match.group(2)},{match.group(3)},{match.group(4)},{match.group(5)},{match.group(6)}\n")
                #print(match.group(3))
            else:
                pass
                #print(line)
                #print("NO MATCH!!!")
                #exit(1)

    return new_path

def fetchPidList( is_aida, path ):
    pids = []
    if is_aida:
        filename = f"{path}/last_pids_aida.txt"
    else:
        filename = f"{path}/last_pids_bench.txt"

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                pids.append(int(line))

    return pids

def formPidList( list_unique, index ):
    last_pid = 0
    curr_index = 0
    res_list = []
    #if index == 0:
    #    return [1461906]
    #if index == 1:
    #    return [1461967]

    for pid in list_unique:
        if last_pid == 0 or abs(pid - last_pid) < 5:
            res_list.append(pid)
        else:
            res_list = [pid]
        
        last_pid = pid 

        if len(res_list) == 4:
            if curr_index == index:
                return res_list
            else:
                res_list = [pid]
                curr_index += 1
    return []

def formDataFrame(csv_path, path, is_aida):
    data = pd.read_csv(csv_path)

    df = pd.DataFrame(data)

     # Convert the timestamps from microseconds to milliseconds (1 ms = 1000 µs)
    df['timestamp'] = df['time'] // 1

    # convert run time from milliseconds to seconds
    #df['run_time'] = df['run_time'] / 1000

    # Calculate the offset from the first timestamp (in milliseconds)
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

    #sorted_unique_tasks = sorted(df["task"].unique())

    pid_list = fetchPidList( is_aida, path )

    print(pid_list)
    
    df = df[df['task'].isin(pid_list)]

    num_events = len( df["timestamp"] )
    mean_run = df["run_time"].mean()

    print(num_events)
    print(mean_run)

    #df2 = df[df["timestamp"] == 60.0]
    #print(len(df2))

    df = df.groupby('timestamp', as_index=False).sum()

    #df_agg = df_agg[df_agg['timestamp'] > 20] 

    return df, num_events, mean_run


bench = "YCSB"
subject = "CPU run time per sec (4 clients on cpus 0-3, benchbase on 4-7)"
path = "/home/carle/Documents/flame/Flamegraph"
filename = "udf_ycsb" # without the extension


csv_path = genCSV(path, filename)

df_aida, num_events_aida, mean_run_aida = formDataFrame(csv_path, path, True)
df_bench, num_events_bench, mean_run_bench = formDataFrame(csv_path, path, False)

#if len(df_aida) == 0:
#    print("AIDA DF empty!!!")
#    exit(1)

if len(df_bench) == 0:
    print("bench DF empty!!!")
    exit(1)

keyword_aida = "w 1 worker (UDF YCSB)"
keyword_bench = f"w 10k worker (bench YCSB)"

key = "run_time"
delay_key = "sch_delay"
wait_key = "wait_time"

fig, ax1 = plt.subplots(figsize=(16, 9))

# Plot latency time series
plt.plot(df_aida['timestamp'], df_aida[key], linewidth=2, label=f"{keyword_aida}", color='red')

plt.plot(df_bench['timestamp'], df_bench[key], linestyle=':', linewidth=2, label=f"{keyword_bench}", color='green')

plt.xlabel('Time (s)', fontsize=18, labelpad=20)  
plt.ylabel('CPU run time (ms)', fontsize=18)


data = [
    [f"{keyword_aida}", f"{df_aida[key].mean():.2f}", f"{num_events_aida}", f"{mean_run_aida:.2f}", f"{df_aida[wait_key].mean():.2f} / {df_aida[delay_key].mean():.2f}"],
    [f"{keyword_bench}", f"{df_bench[key].mean():.2f}", f"{num_events_bench}", f"{mean_run_bench:.2f}", f"{df_bench[wait_key].mean():.2f} / {df_bench[delay_key].mean():.2f}"],
]

# Column labels
columns = ['Process group', 'Mean trs (ms)', 'Run time events', 'Mean run time (ms)', 'Sch delay / wait time (ms)']
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title(f"{subject} - {bench} Benchmark", fontsize=22)

ax1.grid()
plt.legend(fontsize=18)

table_ax = fig.add_axes([0.05, 0.15, 0.9, 0.2])  # Adjust y-position (0.1) and height (0.2)
table_ax.axis('off')  # Turn off the axes for the table

table = table_ax.table(cellText=data, colLabels=columns)

# Set the font size of the table and make the header bold
table.auto_set_font_size(False)
#table.set_fontsize(18)  # Increase the font size for the table

# Make the header bold
for (i, j), cell in table.get_celld().items():
    if j == 4 or j == 0:
        cell.set_width(0.25)
    else:
        cell.set_width(0.19)

    if i == 0:  # Row 0 is the header row
        #cell.set_fontweight('bold')  # Set the header cells to bold
        cell.set_height(0.18)  # Adjust the height of the rows
        cell.set_text_props(weight='bold', fontsize=16)  # Bold and larger header font
    else:
        cell.set_height(0.15)  # Adjust the height of the rows
        cell.set_text_props(fontsize=16)  # Bold and larger header font

plt.savefig(f"hist/{bench}{subject}_test.pdf", format="pdf")

os.system(f"brave \"hist/{bench}{subject}_test.pdf\"")
#plt.show()
# 353319 353364