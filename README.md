Reproducible material for **DW0118:Propagating the prior from far to near offset: A self-supervised diffusion framework for progressively recovering near-offsets of towed-streamer data - Shijun Cheng and Tariq Alkhalifah.**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0118?csf=1&web=1&e=mjnt4P) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* **ssldiff_interpolation**: python library containing routines for self-supervised diffusion-based near-offset reconstruction;
* **dataset**: folder to store dataset;


## Supplementary files
To ensure reproducibility, we provide the the synthetic dataset for training and testing stages and our trainined model. Field data is not shared here due to restricted permissions.

* **Training and testing synthetic data set**
Download the training and testing data set [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0118/dataset.zip?csf=1&web=1&e=EBV3qw). Then, use `unzip` to extract the contents to `dataset/`.

* **Trained model**
Download our trained model [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0118/trained_model.pt?csf=1&web=1&e=oQbS2q). Then, extract the contents to `/checkpoints/`.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate ssldiff-interpolation
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Running code :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can run the training and inference code. 
For traning, you can directly run:
```
python train.py
```

For inference, you can use the synthetic test data we provide and directly run:
```
python sample.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX8000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `ssldiff_interpolation/train.py` file.

## Acknowledgements
This implementation is motivated from the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open source code.

## Cite us 
DW0118 - Cheng and Alkhalifah. (2025) Propagating the prior from far to near offset: A self-supervised diffusion framework for progressively recovering near-offsets of towed-streamer data.

