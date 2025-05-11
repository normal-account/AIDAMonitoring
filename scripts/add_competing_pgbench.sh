/home/build/postgres/start_db.sh

sleep 3

echo Starting good pgbench...
/home/build/run_pgbench.sh
sleep 1
pgrep -f "127.0.0.1\(" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/bench/cgroup.procs; done

echo Starting bad pgbench...
/home/build/run_pgbench.sh
sleep 1
pgrep -f "127.0.0.1\(" | while read pid; do
    current_cgroup=$(cat /proc/$pid/cgroup | awk -F: '{print $3}')
    if [[ "$current_cgroup" != "/parent/bench" ]]; then
        echo $pid | sudo tee /sys/fs/cgroup/parent/aida/cgroup.procs
    fi
done
