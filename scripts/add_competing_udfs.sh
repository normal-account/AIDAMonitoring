/home/build/postgres/start_db.sh

sleep 3

echo Starting ycsb...
/home/build/benchbase/startup_udf_ycsb.sh bixi bixi
sleep 1
pgrep -f "bixi bixi \[local\]" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/bench/cgroup.procs; done

echo Starting tpch...
/home/build/benchbase/startup_udf_tpch.sh bixi bixi
sleep 1
pgrep -f "bixi bixi \[local\]" | while read pid; do
    current_cgroup=$(cat /proc/$pid/cgroup | awk -F: '{print $3}')
    if [[ "$current_cgroup" != "/parent/bench" ]]; then
        echo $pid | sudo tee /sys/fs/cgroup/parent/aida/cgroup.procs
    fi
done
