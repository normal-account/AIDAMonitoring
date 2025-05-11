sudo mkdir /sys/fs/cgroup/parent
sudo mkdir /sys/fs/cgroup/parent/aida
sudo mkdir /sys/fs/cgroup/parent/bench
sudo mkdir /sys/fs/cgroup/parent/java

echo +cpu +io +cpuset +memory | sudo tee /sys/fs/cgroup/parent/cgroup.subtree_control

echo 1 | sudo tee /sys/fs/cgroup/parent/aida/cpu.weight
echo 10000 | sudo tee /sys/fs/cgroup/parent/bench/cpu.weight
echo 10000 | sudo tee /sys/fs/cgroup/parent/java/cpu.weight

echo 0-6 | sudo tee /sys/fs/cgroup/parent/aida/cpuset.cpus
echo 0-6 | sudo tee /sys/fs/cgroup/parent/bench/cpuset.cpus
echo 7 | sudo tee /sys/fs/cgroup/parent/java/cpuset.cpus
