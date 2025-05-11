pgrep -f stress | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/stress/cgroup.procs; done
