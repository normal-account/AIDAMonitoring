i=0

pgrep -f "burn_cpu" | while read pid; do i=$(($i+1)); echo lw$i : $pid; echo $pid | sudo tee /sys/fs/cgroup/parent/lw$i/cgroup.procs; done