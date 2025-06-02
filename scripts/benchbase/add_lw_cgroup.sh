pgrep -f "\[local\]" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/lw/cgroup.procs; done
