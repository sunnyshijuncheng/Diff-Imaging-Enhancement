![LOGO](https://github.com/sunnyshijuncheng/Diff-Imaging-Enhancement/blob/main/asset/logo.jpg)

Reproducible material for Generative Diffusion Model for Seismic Imaging Improvement of Sparsely Acquired Data and Uncertainty Quantification - Xingchen Shi, Shijun Cheng, Weijian Mao, and Wei Ouyang.

# Project structure
This repository is organized as follows:

* :open_file_folder: **diff_imaging_enhancement**: python code containing routines for Generative Diffusion Model for Seismic Imageing Enhancement;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset;
* :open_file_folder: **scripts**: set of shell scripts used to run training and testing

## Supplementary files
To ensure reproducibility, we provide the the data set for training and testing, and the trained model for validate the low-frequency extrapolation performance. 

* **Training and Testing data set**
Download the training and testing data set [here](https://drive.google.com/file/d/1VXcms_hpkMde7k41iuODtF33pOwHaU6b/view?usp=drive_link). Then, use `unzip` to extract the contents to `dataset/train` and `dataset/test`.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate  diff-image-enhancement
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Scripts :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can entry the scripts file folder and run demo. We provide two scripts which are responsible for meta-train and meta-test examples.

For training, you can directly run:
```
sh run_train.sh
```

For testing, you can directly run:
```
sh run_test.sh
```

If you need to compare with a randomly initialized network, you can comment out lines 63 and 64 in the `metaprocessing/meta_test/train.py` file as follows
```
# net.load_state_dict(torch.load(dir_load, map_location=device))
# print(f'Model loaded from {dir_load}')
```
and then run:
```
sh run_meta_test.sh
```

**Note:** We emphasize that the training logs (for both meta-train and meta-test) is saved in the `runs/` file folder. You can use the `tensorboard --logdir=./` or extract the log to view the changes of the metrics as a function of epoch.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. Due to the high memory consumption during the training phase, if your graphics card does not support large batch training, please reduce the configuration value of args (`batch_size`)

## Cite us 
```bibtex
@article{shi2024generative,
  title={Generative Diffusion Model for Seismic Imaging Improvement of Sparsely Acquired Data and Uncertainty Quantification},
  author={Shi, Xingchen and Cheng, Shijun and Mao, Weijian and Ouyang, Wei},
  journal={arXiv preprint arXiv:2407.21683},
  year={2024}
}

