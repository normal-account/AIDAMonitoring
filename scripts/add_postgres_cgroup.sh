pgrep -f "postgres" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/java/cgroup.procs; done
