<div align="center">

<h3><strong>Propagating the prior from far to near offset: A self-supervised diffusion framework for progressively recovering near offsets of towed-streamer data</strong></h3>

</div>

## Overview

This repository provides the official implementation of a self-supervised generative diffusion model for reconstructing missing near-offset traces in marine towed-streamer seismic data.

The proposed method constructs self-supervised training pairs directly from the available recorded data. Two overlapping seismic patches are extracted with a one-trace lateral shift. One patch is used as the clean target, while the adjacent patch provides the conditioning information.

During inference, the trained conditional diffusion model recursively predicts a patch shifted by one trace toward the missing near-offset region. The newly reconstructed patch is then used as the conditioning input for the next recursive step, progressively propagating the learned prior information from the recorded far-offset region toward zero offset.

The generative formulation also supports uncertainty quantification. Multiple reconstructions are generated using different random-noise initializations. Their ensemble mean is used as the final reconstruction, while their pointwise standard deviation provides an uncertainty estimate.

---

## Project structure

This repository is organized as follows:

```text
repository-root/
├── dataset/                         # Directory for the downloaded seismic datasets
├── ssldiff_interpolation/
│   ├── code/                        # Core diffusion-model library
│   ├── sample.py                    # Sampling and near-offset reconstruction script
│   └── train.py                     # Model-training script
├── environment.yml                  # Conda environment configuration
├── install_env.sh                   # Environment installation script
├── LICENSE                          # Repository license
└── README.md
```

Main components:

- :open_file_folder: **`dataset`**: Directory for storing the SEAM and Viking datasets downloaded from Zenodo.
- :open_file_folder: **`ssldiff_interpolation/code`**: Python library containing the dataset loader, conditional U-Net, Gaussian diffusion process, timestep samplers, timestep respacing utilities, and training loop.
- :page_facing_up: **`ssldiff_interpolation/train.py`**: Example script for training the proposed model on the SEAM dataset.
- :page_facing_up: **`ssldiff_interpolation/sample.py`**: Example script for recursive near-offset reconstruction and uncertainty quantification on the SEAM dataset.
- :page_facing_up: **`environment.yml`**: Conda environment specification.
- :page_facing_up: **`install_env.sh`**: Shell script for creating the required Conda environment.

The main files in `ssldiff_interpolation/code` include:

```text
code/
├── datasets.py               # Self-supervised seismic patch construction
├── gaussian_diffusion.py     # Diffusion training and sampling procedures
├── resample.py               # Diffusion-timestep sampling strategies
├── respace.py                # Timestep respacing for accelerated sampling
├── script_util.py            # Model and diffusion configuration utilities
├── train_util.py             # Model-training loop
└── unet.py                   # Conditional U-Net architecture
```

---

## Supplementary files

To support reproducibility, the datasets and trained models used in the experiments are provided through Zenodo:

**Zenodo DOI:** https://doi.org/10.5281/zenodo.21325738

The Zenodo record contains two compressed files:

```text
dataset.zip
trained_model.zip
```

The first field dataset used in the manuscript cannot be publicly distributed because of data-access restrictions.

### Datasets

The file `dataset.zip` contains the SEAM and Mobil AVO Viking Graben Line 12 datasets used in the numerical experiments.

After extracting `dataset.zip`, the directory structure is:

```text
dataset/
├── seam/
│   ├── train/
│   └── train_rotated/
└── viking/
    ├── train/
    └── train_rotated/
```

All seismic shot gathers are stored in MATLAB `.mat` format.

The folders have the following meanings:

- **`train`**: Original seismic shot gathers used to train the proposed self-supervised diffusion model.
- **`train_rotated`**: Rotated seismic shot gathers used to train the self-supervised rotation-truncation baseline following Wang et al. The rotation procedure is described in the manuscript.

The SEAM dataset is used for the synthetic controlled validation experiment.

The Viking dataset is used for:

- the controlled validation experiment, in which additional recorded traces are artificially removed and retained as references; and
- the real-world application, in which the actual acquisition-related near-offset gap is reconstructed without ground-truth near-offset data.

The first field dataset described in the manuscript is not included because its distribution is restricted.

### Trained models

The file `trained_model.zip` contains three pretrained model checkpoints:

```text
model_seam.pt
model_viking_control.pt
model_viking_real.pt
```

The checkpoints correspond to the following experiments:

- **`model_seam.pt`**: Model trained on the SEAM dataset for the controlled validation experiment.
- **`model_viking_control.pt`**: Model trained on the Viking dataset for the controlled validation experiment, where additional recorded traces are artificially removed for quantitative evaluation.
- **`model_viking_real.pt`**: Model trained on the complete observed Viking dataset for reconstructing the actual acquisition-related near-offset gap.

After extracting `trained_model.zip`, place the required model checkpoint in a convenient directory and update the checkpoint path in `sample.py` accordingly.

---

## Getting started :space_invader: :robot:

We recommend creating the Conda environment using the provided `environment.yml` file.

From the repository root directory, run:

```bash
./install_env.sh
```

The installation may take some time. If `Done!` appears in the terminal at the end of the installation, the environment has been successfully created.

Activate the environment using:

```bash
conda activate ssldiff-interpolation
```

No additional package installation is required if the scripts are run from the `ssldiff_interpolation` directory as described below.

---

## Preparing the supplementary files

Download `dataset.zip` and `trained_model.zip` from:

https://doi.org/10.5281/zenodo.21325738

Extract `dataset.zip` into the repository root directory so that the resulting structure is:

```text
repository-root/
├── dataset/
│   ├── seam/
│   │   ├── train/
│   │   └── train_rotated/
│   └── viking/
│       ├── train/
│       └── train_rotated/
└── ssldiff_interpolation/
```

Extract `trained_model.zip` and place the model files in a directory accessible to `sample.py`, for example:

```text
repository-root/
├── trained_model/
│   ├── model_seam.pt
│   ├── model_viking_control.pt
│   └── model_viking_real.pt
└── ssldiff_interpolation/
```

The model path in `sample.py` should be updated to point to the selected checkpoint.

---

## Running the code :page_facing_up:

The provided `train.py` and `sample.py` scripts demonstrate the SEAM controlled validation experiment.

Because the scripts use relative dataset paths, first move into the source-code directory:

```bash
cd ssldiff_interpolation
```

### Training

To train the proposed self-supervised diffusion model on the SEAM dataset, run:

```bash
python train.py
```

The default configuration includes:

```text
Training dataset:       ../dataset/seam/train/
Batch size:             16
Patch width:            32 traces
Learning rate:          1e-4
Diffusion steps:        1000
Noise schedule:         cosine
Prediction target:      clean target patch x_0
EMA rate:               0.999
Training iterations:    20000
```

During training, two overlapping patches are extracted from each shot gather with a one-trace lateral shift.

The conditional diffusion model receives:

- a noisy version of the target patch, denoted by `x_t`; and
- the adjacent conditioning patch, denoted by `shot_far`.

The U-Net is trained to directly predict the clean target patch `x_0`.

The default network configuration includes:

```text
Input channels:          2
Output channels:         1
Base channels:           64
Channel multipliers:     1, 2, 4, 8, 16
Residual blocks/scale:   2
Attention heads:         4
Attention resolutions:   8 and 16
```

Trained EMA checkpoints are saved in the checkpoint directory specified in the training utilities.

### Sampling and inference

To run the SEAM controlled reconstruction example, use:

```bash
python sample.py
```

Before running the script, update the model checkpoint path in `sample.py` so that it points to:

```text
model_seam.pt
```

The sampling script performs recursive near-offset reconstruction.

At each recursive step:

1. The current recorded or reconstructed patch is used as the conditioning input.
2. The diffusion model generates a patch shifted by one trace toward the missing region.
3. The generated patch becomes the conditioning input for the next recursive step.
4. The process continues until all designated missing traces have been reconstructed.

---

## DDIM and DDPM sampling

The sampling method is controlled by:

```text
use_ddim
timestep_respacing
```

To use accelerated DDIM sampling, set:

```bash
python sample.py --use_ddim True --timestep_respacing ddim2
```

Here, `ddim2` selects two timesteps from the original 1000-step diffusion process.

Other DDIM configurations can also be used, for example:

```bash
python sample.py --use_ddim True --timestep_respacing ddim10
```

To use the full DDPM reverse-sampling process, run:

```bash
python sample.py --use_ddim False
```

The experiments in the manuscript mainly use accelerated DDIM sampling for recursive near-offset reconstruction.

---

## Ensemble uncertainty quantification

In `sample.py`, `batch_size` represents the number of independent diffusion realizations generated from different random-noise initializations.

For example:

```bash
python sample.py --batch_size 20
```

generates 20 reconstruction realizations.

The script calculates:

- the ensemble mean as the final reconstructed seismic patch; and
- the pointwise standard deviation as the uncertainty estimate.

The output MATLAB file contains:

```text
predict
uq
accs
```

where:

- **`predict`** is the ensemble-mean reconstructed seismic patch;
- **`uq`** is the pointwise standard deviation across the diffusion realizations;
- **`accs`** is the mean-squared reconstruction error for the controlled experiment, where reference traces are available.

For real-world applications, ground-truth near-offset traces are unavailable. In that case, `uq` provides an indicator of reconstruction confidence.

---

## Viking experiments

The provided `train.py` and `sample.py` scripts demonstrate the SEAM experiment by default.

To reproduce the Viking experiments, update the following settings according to the configurations described in the manuscript:

- dataset path;
- model checkpoint path;
- number of missing traces;
- number of recorded traces used for training;
- temporal truncation;
- number of training iterations;
- recursive reconstruction range; and
- DDIM sampling configuration.

Use:

```text
model_viking_control.pt
```

for the controlled Viking experiment.

Use:

```text
model_viking_real.pt
```

for reconstruction of the actual acquisition-related near-offset gap.

### Controlled Viking experiment

In the controlled experiment, additional recorded traces are artificially removed from the observed data. These traces are retained as references for quantitative evaluation.

The model checkpoint used for this experiment is:

```text
model_viking_control.pt
```

### Real-world Viking application

In the real-world application, the model is trained using the complete observed Viking data and reconstructs the actual near-offset gap caused by the physical source-receiver separation.

The model checkpoint used for this experiment is:

```text
model_viking_real.pt
```

Because the true near-offset traces were never recorded, direct error calculation is impossible. The ensemble uncertainty map is therefore used as the primary reconstruction-confidence indicator.

---

## Baseline data

The `train_rotated` folders contain the rotated shot gathers used for the self-supervised rotation-truncation baseline following Wang et al.

The rotated data are provided for reproducing the baseline comparison described in the manuscript.

The proposed diffusion model uses the data in:

```text
train/
```

whereas the rotation-truncation baseline uses the corresponding data in:

```text
train_rotated/
```

The baseline training and evaluation settings should follow those described in the manuscript.

---

## Reproducibility workflow

A typical workflow for reproducing the SEAM controlled experiment is as follows.

1. Clone or download this repository.

2. Create and activate the Conda environment:

```bash
./install_env.sh
conda activate ssldiff-interpolation
```

3. Download the supplementary files from Zenodo:

https://doi.org/10.5281/zenodo.21325738

4. Extract `dataset.zip` into the repository root directory.

5. Extract `trained_model.zip` into a local model directory.

6. Move into the source-code directory:

```bash
cd ssldiff_interpolation
```

7. To retrain the SEAM model, run:

```bash
python train.py
```

8. To reproduce the SEAM reconstruction using the provided model, update the model path in `sample.py` to point to `model_seam.pt`.

9. Run inference:

```bash
python sample.py
```

10. Open the generated MATLAB output file to examine:

```text
predict
uq
accs
```

---

## Hardware and environment

The experiments were conducted on a workstation equipped with an Intel(R) Xeon(R) CPU @ 2.10 GHz and a single NVIDIA GeForce RTX 8000 GPU.

Different hardware and software configurations may require minor adjustments.

If the available GPU memory is insufficient for the default training configuration, reduce the `batch_size` argument in:

```text
ssldiff_interpolation/train.py
```

For inference, reducing `batch_size` decreases the number of ensemble realizations and GPU-memory consumption. However, using fewer realizations may reduce the stability of the estimated ensemble mean and uncertainty map.

---

## Data availability

The SEAM and Mobil AVO Viking Graben Line 12 datasets used in the provided reproducibility examples are available through the accompanying Zenodo record:

https://doi.org/10.5281/zenodo.21325738

The first field dataset used in the manuscript is subject to access restrictions and cannot be publicly distributed.

---

## Acknowledgements

This implementation is based in part on the diffusion-model framework introduced in:

> Nichol, A. Q., and P. Dhariwal, 2021, Improved denoising diffusion probabilistic models.

The Gaussian diffusion utilities and U-Net implementation were adapted from the open-source `improved-diffusion` repository developed by OpenAI:

https://github.com/openai/improved-diffusion

We gratefully acknowledge the authors for making their implementation publicly available.

---

## License

Please refer to the `LICENSE` file included in this repository for the applicable usage and distribution terms.
