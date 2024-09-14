"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.gethostname())

    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    # 如果 CUDA 可用，函数返回一个 PyTorch 设备，指定为 GPU。
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    # 如果 CUDA 不可用，则函数返回一个指向 CPU 的 PyTorch 设备。
    return th.device("cpu")

'''
这个 load_state_dict 函数是 dist_util 模块的一部分，用于在分布式训练环境中加载 PyTorch 模型的状态字典。
这个函数特别设计来避免在 MPI的每个进程中重复读取相同的文件，从而提高效率。
检查进程排名:
if MPI.COMM_WORLD.Get_rank() == 0:
检查当前进程是否是主进程（rank 0）。在分布式训练中，通常只有主进程（rank 0）执行某些操作，如读取文件。
主进程中读取文件:
with bf.BlobFile(path, "rb") as f:
在主进程中，使用 bf.BlobFile 打开状态字典文件进行读取。这里 path 是文件路径，"rb" 模式表示以二进制形式读取。
data = f.read() 读取整个文件内容到变量 data。
非主进程设置 data 为空:
data = MPI.COMM_WORLD.bcast(data): 将主进程中读取的数据 data 发送到所有其他进程。
使用 PyTorch 的 th.load 函数从字节流中加载状态字典。io.BytesIO(data) 创建了一个从 data 读取的字节流对象，
**kwargs 允许传递额外的参数给 th.load 函数（如模型的设备信息）。
'''
def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)

'''
用于在分布式训练环境中同步参数。它确保了所有进程中的参数与主进程（rank 0）中的参数保持一致。
dist.broadcast(p, 0)
使用 PyTorch 的 torch.distributed.broadcast 函数广播参数。
这个函数将参数 p 从源进程（在这里是 rank 0，即主进程）广播到所有其他进程。
broadcast 函数确保所有进程中的相应参数 p 都与源进程中的参数相同。
这在初始化分布式训练时或在从检查点恢复时同步参数时非常重要。
'''
def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
