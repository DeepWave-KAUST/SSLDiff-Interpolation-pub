import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch

class CUDAPrefetcher():
    """CUDA prefetcher.
    Ref:
    https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt=None):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            if type(self.batch) == dict:
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            elif type(self.batch) == list:
                for k in range(len(self.batch)):
                    if torch.is_tensor(self.batch[k]):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            else:
                assert NotImplementedError

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()

def load_data(
    *, data_dir, batch_size, width_size, device, class_cond=False, deterministic=False
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
        width_size, 
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

class BasicDataset(Dataset):
    def __init__(self, paths, width_size, class_cond=False):
        super().__init__()
        self.local_dataset = paths
        self.class_cond = class_cond
        self.width_size = width_size

    def __len__(self):
        return len(self.local_dataset)

    def __getitem__(self, idx):
        path = self.local_dataset[idx]

        dict = sio.loadmat(path)
        shot = dict['shot']
        nt, nr_ori = shot.shape
        width_gap = 1
        miss_trace = 10
        nr_max = nr_ori - miss_trace - width_gap
        nr_min = nr_ori // 2

        width_far = random.randint(nr_min, nr_max - self.width_size)
        width_near = width_far + width_gap
        shot_far = shot[:, width_far:width_far+self.width_size]
        shot_near = shot[:, width_near:width_near+self.width_size]

        shot_far = np.expand_dims(np.array(shot_far, dtype=np.float32), axis=0)
        shot_near = np.expand_dims(np.array(shot_near, dtype=np.float32), axis=0)

        out_dict = {}
        return shot_near, shot_far, out_dict
