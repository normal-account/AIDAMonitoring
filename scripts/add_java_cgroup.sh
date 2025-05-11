pgrep -f "java -jar benchbase" | while read pid; do echo $pid | sudo tee /sys/fs/cgroup/parent/java/cgroup.procs; done
