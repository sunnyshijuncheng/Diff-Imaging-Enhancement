from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio

def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    dataset = BasicDataset(
        all_files,
        class_cond=class_cond,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

'''
shard参数通过调用MPI库的 Get_rank() 函数获取。Get_rank() 返回当前进程的“排名”或“ID”。
num_shards参数通过调用 MPI 的 Get_size() 函数获取。Get_size() 返回在当前 MPI 通信域中的进程总数。
如果您的环境中有两个GPU, MPI 会给每个进程分配一个唯一的排名（rank），通常是从 0 开始。
因此，在两个 GPU 的情况下，这些排名将是 0 和 1。
进程总数num_shards则为2
因此在调取数据时，image_paths[shard:][::num_shards]
GPU0会从第0个数据开始，每2个数据取一个
GPU1会从第1个数据开始，每2个数据取一个
这样两个GPU各取一半数据
'''
class BasicDataset(Dataset):
    def __init__(self, paths, class_cond=False):
        super().__init__()
        # paths[shard:][::num_shards] 首先截取从 shard 索引开始的所有元素，
        # 然后从这些元素中以 num_shards 为步长选取元素。
        self.local_dataset = paths
        self.class_cond = class_cond

    def __len__(self):
        return len(self.local_dataset)

    def __getitem__(self, idx):
        path = self.local_dataset[idx]

        dict = sio.loadmat(path)
        sparse_image = dict['sparse']
        dense_image = dict['dense']

        sparse_image = np.array(sparse_image, dtype=np.float32)
        dense_image = np.array(dense_image, dtype=np.float32)

        out_dict = {}
        return np.expand_dims(dense_image, axis=0), np.expand_dims(sparse_image, axis=0), out_dict
