# 1) Create namespaces
ip netns add ns_mlx5_4
ip netns add ns_mlx5_5

# 2) Move interfaces into separate namespaces
ip link set ens4013f0np0 netns ns_mlx5_4
ip link set ens4013f1np1 netns ns_mlx5_5

# 3) Bring up loopback and NICs
ip -n ns_mlx5_4 link set lo up
ip -n ns_mlx5_5 link set lo up
ip -n ns_mlx5_4 link set ens4013f0np0 up
ip -n ns_mlx5_5 link set ens4013f1np1 up

# 4) Assign point-to-point test subnet
ip -n ns_mlx5_4 addr add 192.168.40.1/30 dev ens4013f0np0
ip -n ns_mlx5_5 addr add 192.168.40.2/30 dev ens4013f1np1

ip netns exec ns_mlx5_4 ping -c 3 192.168.40.2
ip netns exec ns_mlx5_5 ping -c 3 192.168.40.1

ip netns exec ns_mlx5_4 ib_write_bw -d mlx5_4 -i 1 -x 3 -p 19600 -s 4096 -n 1000 &
sleep 1
ip netns exec ns_mlx5_5 ib_write_bw -d mlx5_5 -i 1 -x 3 -p 19600 -s 4096 -n 1000 192.168.40.1


pkill -f ib_write_bw || true
ip -n ns_mlx5_4 link set ens4013f0np0 netns 1
ip -n ns_mlx5_5 link set ens4013f1np1 netns 1
ip netns del ns_mlx5_4
ip netns del ns_mlx5_5