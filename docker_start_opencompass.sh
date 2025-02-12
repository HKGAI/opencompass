PROJECT_ROOT=/aifs4su/code
DOCKER_IMG=registry-intl.cn-hongkong.aliyuncs.com/sixpublic/pytorch:23.10-py3


docker run --name opencompass_2 --net=host --ipc=host --rm -it --gpus all --shm-size=1024g --ulimit memlock=-1 --privileged \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e NCCL_SOCKET_IFNAME=ibp \
     -e NCCL_IB_HCA=mlx5 \
     -e NCCL_DEBUG_SUBSYS=ALL \
     -e MASTER_PORT=6000 \
     -v $PROJECT_ROOT/opencompass:/workspace/opencompass \
     -v $PROJECT_ROOT/opencompass_new:/workspace/opencompass_new \
     -v $PROJECT_ROOT:/workspace/code \
     -v /run/mellanox/drivers:/run/mellanox/drivers:shared \
     -v /etc/network:/etc/network \
     -v /etc:/host/etc \
     -v /lib/udev:/host/lib/udev \
     -v $PROJECT_ROOT/checkpoints:/workspace/checkpoints \
     -w /workspace/opencompass \
     $DOCKER_IMG
