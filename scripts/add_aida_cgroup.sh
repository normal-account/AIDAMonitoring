pgrep -f "bixi bixi \[local\]" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/aida/cgroup.procs; done
