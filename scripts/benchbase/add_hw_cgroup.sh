KEYWORD="127.0.0.1\("

pgrep -f "$KEYWORD" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/hw/cgroup.procs; done
