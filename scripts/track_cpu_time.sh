#!/usr/bin/env bash

# Usage: ./track_cpu_time.sh PID1 PID2 ...
# Tracks approximate time each PID spends on each CPU.

declare -A cpu_time

#keyword="127.0.0.1"
keyword="intermittent"
#keyword="burn_cpu"

pid_list=$(pgrep -f $keyword | tr "\n" " ")

pgrep -f $keyword  > /dev/null
if [ $? -ne 0 ]; then
        echo Found no pids. Exiting.
        exit 1
fi

# Initialize associative arrays for each PID
for pid in "$pid_list"; do
    for cpu in $(seq 0 "$(nproc --all)"); do
        cpu_time["$pid,$cpu"]=0
    done
done

# Sampling interval in seconds
interval=0.1

echo "Tracking PIDs $pid_list..."

while [ $? -eq 0 ]; do
    for pid in $pid_list; do
            # Field 39 in /proc/<pid>/stat is the last CPU the process was on
            cpu=$(awk '{print $39}' /proc/$pid/stat 2> /dev/null)
            key="$pid,$cpu"
            cpu_time["$key"]=$((${cpu_time["$key"]} + 1))
    done
    #sleep $interval
    pgrep -f $keyword &> /dev/null
done

# Trap Ctrl+C and print the results
total_ticks=0

echo -e "\n######################\nPIDs STATS\n######################"

for pid in $pid_list; do
    echo "PID $pid:"
    total_ticks_pid=0
    for cpu in $(seq 0 "$(nproc --all)"); do
            total_ticks_pid=$((total_ticks_pid + cpu_time["$pid,$cpu"]))
    done

    for cpu in $(seq 0 "$(nproc --all)"); do
        ticks=${cpu_time["$pid,$cpu"]}
        if [ "$ticks" ]; then
                portion=$(echo "scale=2; $ticks / $total_ticks_pid * 100.0" | bc)
                printf "  CPU $cpu: ~%.2f ticks (~%.2f%%)\n" "$ticks" "$portion"
        fi
    done
    total_ticks=$((total_ticks+total_ticks_pid))
done;


echo -e "\n\n######################\nCPUS STATS\n######################"

for cpu in $(seq 0 "$(nproc --all)"); do
        total_ticks_cpu=0
        for pid in $pid_list; do
                total_ticks_cpu=$((total_ticks_cpu + cpu_time["$pid,$cpu"]))
        done

        if [ $total_ticks_cpu -ne 0 ]; then
                portion=$(echo "scale=2; $total_ticks_cpu / $total_ticks * 100.0" | bc)
                printf "  CPU $cpu: ~%.2f ticks (~%.2f%%)\n" "$total_ticks_cpu" "$portion"
        fi
done

echo -e "\n"

./kill_burns.sh