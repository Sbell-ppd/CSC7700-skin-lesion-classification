# CSC-7333-Skin-Cancer-Classification

## Setup 

```shell
# clone repo
$ git clone https://github.com/Fakorede/CSC-7333-Skin-Cancer-Classification.git
$ cd CSC-7333-Skin-Cancer-Classification

# create environment
$ conda create -n skin-env python=3.10 -y
$ conda activate skin-env

# install dependencies
$ pip install -r requirements.txt
```

## Folder Structure

```
project_root/
├── data/                               # dataset in standard image classification format
├── models/                             # trained model parameters
├── data_preprocessing.ipynb            # notebok for processing data
├── download_data.py                    # script for downloading data
├── exploratory-data-analysis.ipynb     # notebook for exploring data
├── main.py                             # functions to train&test
├── model.py                            # build models
├── train.py                            # train PyTorch models
├── SkinDataset.py                      # transform the images into tensors
└── utils.py                            # utility functions
```

## Run 


```shell
# download dataset
$ python3 download_data.py

# play with notebooks
$ jupyter notebook

# sample command to train model
$ python3 main.py --model resnet --batch-size 32 --lr 0.001 --num-epochs 10
```
