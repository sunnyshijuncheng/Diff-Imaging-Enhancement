![LOGO](https://github.com/DeepWave-Kaust/Meta-Processing/blob/main/asset/logo.jpg)

Reproducible material for Generative Diffusion Model for Seismic Imaging Improvement of Sparsely Acquired Data and Uncertainty Quantification - Xingchen Shi, Shijun Cheng, Weijian Mao, and Wei Ouyang.

# Project structure
This repository is organized as follows:

* :open_file_folder: **diff_imaging_enhancement**: python code containing routines for Meta-Processing, which include two parts: meta-train and meta-test;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset;
* :open_file_folder: **results**: folder to store meta-initialization neural network model;
* :open_file_folder: **scripts**: set of python scripts for reproducing the meta-train and meta-test examples


## Supplementary files
To ensure reproducibility, we provide the the data set for meta-train and meta-test stages, and the meta-initialization model for various seismic processing. In meta-train stage, we just include **Note:** If you wish to train the models from random initialization, please **do not** download and copy the meta-initialization models.

* **Meta-train data set**
Download the meta-train data set [here](https://drive.google.com/drive/folders/1JyWOVd6ohIQR7Yw8Qf5DxIUBMFotOKmm?usp=sharing). Then, use `unzip` to extract the contents to `meta_train_dataset/`.

* **Meta-test data set**
Download the meta-test data set [here](https://drive.google.com/drive/folders/19FZB8brT0zH-ccgH_M5ZzEg0BCIKSSK8?usp=sharing). Then, use `unzip` to extract the contents to `meta_test_dataset/`.

* **Meta-initialization model**
Download the meta-initialization neural network model [here](https://drive.google.com/drive/folders/1u5rHqHBiXxxq38UPEOFPYG7DoVOlhk5b?usp=sharing). Then, extract the contents to `meta_checkpoints/`.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate  meta-processing
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

For meta-train, you can directly run:
```
sh run_meta_train.sh
```
**Note:** When you run demo for meta-train, you need open the `metaprocessing/meta_train/train.py` file to modify the meta-train dataset file folder accordingly.

For meta-test, you can directly run:
```
sh run_meta_test.sh
```
**Note:** When you run demo for meta-test, you need open the `metaprocessing/meta_test/train.py` file to modify the meta-test dataset file folder accordingly, which depends on the seismic processing task you want to test. Meanwhile, you need open the `metaprocessing/meta_test/train.py` file to specify the path for meta initialization model. Here, we have provided a meta-initialization model in supplementary file, you can directly load meta-initialization model to perform meta-test.

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
configurations may be required for different combinations of workstation and GPU. Due to the high memory consumption during the meta training phase, if your graphics card does not support large batch training, please reduce the configuration value of args (`args.k_spt` and `args.k_qry`) in the `metaprocessing/meta_train/train.py` file.

## Cite us 
```bibtex
@article{cheng2024meta,
  title={Meta-processing: A robust framework for multi-tasks seismic processing},
  author={Cheng, Shijun and Harsuko, Randy and Alkhalifah, Tariq},
  journal={Surveys in Geophysics},
  pages={1--36},
  year={2024},
  publisher={Springer}
}

