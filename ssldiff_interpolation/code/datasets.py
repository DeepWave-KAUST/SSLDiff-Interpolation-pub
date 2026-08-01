import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch


def load_data(
    *, data_dir, batch_size, width_size, device, class_cond=False, deterministic=False
):
    """
    Create an infinite data generator for self-supervised diffusion training.

    Each iteration returns a batch containing two laterally shifted seismic
    patches and an auxiliary dictionary:

        shot_near:
            The patch used as the diffusion target.

        shot_far:
            The adjacent patch used as the conditioning input.

        out_dict:
            An auxiliary dictionary reserved for optional conditioning
            information, such as class labels.

    The two seismic patches have a one-trace lateral shift, following the
    self-supervised training-pair construction described in Equation (7) of
    the manuscript.

    Parameters
    ----------
    data_dir : str
        Directory containing the seismic shot gathers stored as MATLAB files.

    batch_size : int
        Number of shot-gather patches in each training batch.

    width_size : int
        Lateral width W of each extracted seismic patch.

    device
        Computing device passed through the data-loading interface. The data
        are transferred to the device later in the training workflow.

    class_cond : bool, optional
        Whether class-conditioning information is included in the returned
        dictionary. Class conditioning is not used in the current framework.

    deterministic : bool, optional
        If True, preserve a deterministic dataset order. Otherwise, randomly
        shuffle the shot gathers before forming each epoch.

    Yields
    ------
    tuple
        A batch of target patches, conditioning patches, and auxiliary
        conditioning dictionaries.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Recursively collect all MATLAB files containing seismic shot gathers.
    all_files = _list_image_files_recursively(data_dir)

    # Construct the dataset that generates one-trace-shifted patch pairs.
    dataset = BasicDataset(
        all_files,
        width_size,
        class_cond=class_cond,
    )

    # Create mini-batches for diffusion-model training.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=4,
        drop_last=True
    )

    # Continuously cycle through the dataset during iterative model training.
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively locate MATLAB files in the specified dataset directory.

    Parameters
    ----------
    data_dir : str
        Root directory containing the seismic training data.

    Returns
    -------
    list
        Sorted list of paths to MATLAB files.
    """
    results = []

    # Sort directory entries to provide a stable file-discovery order.
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]

        # Add MATLAB files containing individual seismic shot gathers.
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)

        # Recursively search nested dataset directories.
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))

    return results


class BasicDataset(Dataset):
    """
    Dataset for constructing self-supervised seismic patch pairs.

    For each shot gather, two overlapping patches with the same lateral width
    are extracted. Their starting positions differ by one trace, allowing the
    model to learn the local statistical relationship between adjacent offsets.

    Parameters
    ----------
    paths : list
        Paths to MATLAB files containing seismic shot gathers.

    width_size : int
        Number of traces W included in each target and conditioning patch.

    class_cond : bool, optional
        Flag reserved for optional class conditioning.
    """

    def __init__(self, paths, width_size, class_cond=False):
        super().__init__()

        # Store the paths of all available shot gathers.
        self.local_dataset = paths

        # Preserve the class-conditioning option for compatibility with the
        # diffusion-model data interface.
        self.class_cond = class_cond

        # Lateral patch width W used in the self-supervised training pairs.
        self.width_size = width_size

    def __len__(self):
        """Return the number of available shot gathers."""
        return len(self.local_dataset)

    def __getitem__(self, idx):
        """
        Load one shot gather and construct a self-supervised patch pair.

        The two patches overlap over width_size - 1 traces and are separated
        by a one-trace shift. This construction trains the diffusion model to
        predict an adjacent patch from its neighboring offset context.

        Parameters
        ----------
        idx : int
            Index of the selected shot-gather file.

        Returns
        -------
        shot_near : numpy.ndarray
            Single-channel target patch with shape
            (1, nt, width_size).

        shot_far : numpy.ndarray
            Single-channel conditioning patch with shape
            (1, nt, width_size).

        out_dict : dict
            Empty auxiliary dictionary reserved for additional conditioning
            information.
        """
        path = self.local_dataset[idx]

        # Load the MATLAB file and extract the seismic shot gather.
        dict = sio.loadmat(path)
        shot = dict['shot']

        # nt is the number of temporal samples, and nr_ori is the original
        # number of receiver traces in the shot gather.
        nt, nr_ori = shot.shape

        # Use a one-trace shift between the two overlapping patches, matching
        # the recursive one-trace extrapolation strategy used during inference.
        width_gap = 1

        # Number of near-offset traces treated as missing in the controlled
        # validation experiments.
        miss_trace = 10

        # Define the largest valid starting range while reserving the simulated
        # missing region and the one-trace shift between the two patches.
        nr_max = nr_ori - miss_trace - width_gap

        # Restrict random patch extraction to the selected half of the receiver
        # aperture used by this dataset configuration.
        nr_min = nr_ori // 2

        # Randomly select the starting trace of the first patch. The upper
        # bound ensures that a complete patch of width_size traces can be
        # extracted after accounting for the one-trace shift.
        width_far = random.randint(nr_min, nr_max - self.width_size)

        # Shift the starting position by one receiver trace to construct the
        # adjacent overlapping patch.
        width_near = width_far + width_gap

        # Extract two overlapping patches with a one-trace lateral displacement.
        shot_far = shot[:, width_far:width_far+self.width_size]
        shot_near = shot[:, width_near:width_near+self.width_size]

        # Convert the target patch to single-precision floating point and add
        # the channel dimension required by the 2D convolutional U-Net.
        shot_far = np.expand_dims(
            np.array(shot_far, dtype=np.float32),
            axis=0
        )

        # Convert the conditioning patch to single-precision floating point
        # and add the channel dimension. Each returned sample therefore has
        # the shape (1, nt, width_size).
        shot_near = np.expand_dims(
            np.array(shot_near, dtype=np.float32),
            axis=0
        )

        # Reserve an auxiliary dictionary for optional conditioning variables.
        # The current self-supervised framework does not use class labels.
        out_dict = {}

        # Return the two adjacent patches and optional conditioning metadata.
        return shot_near, shot_far, out_dict
