sudo mkdir /sys/fs/cgroup/parent
sudo mkdir /sys/fs/cgroup/parent/lw
sudo mkdir /sys/fs/cgroup/parent/hw

echo +cpu +io +cpuset +memory | sudo tee /sys/fs/cgroup/parent/cgroup.subtree_control

echo 1 | sudo tee /sys/fs/cgroup/parent/lw/cpu.weight
echo 10000 | sudo tee /sys/fs/cgroup/parent/hw/cpu.weight

# For benchmarks pinning each process to a specific CPU
for i in $(seq 1 $CLIENTS); do
        sudo mkdir /sys/fs/cgroup/parent/hw$i
        sudo mkdir /sys/fs/cgroup/parent/lw$i

        echo $((10000/$CLIENTS)) | sudo tee /sys/fs/cgroup/parent/hw$i/cpu.weight
        echo 1 | sudo tee /sys/fs/cgroup/parent/lw$i/cpu.weight
done

echo 0-$(($CLIENTS-1)) | sudo tee /sys/fs/cgroup/parent/cpuset.cpus