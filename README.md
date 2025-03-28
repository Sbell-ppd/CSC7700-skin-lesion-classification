# CSC-7333-Skin-Cancer-Classification

## Setup 

```shell
# clone repo
$ git clone https://github.com/Fakorede/CSC-7333-Skin-Cancer-Classification.git
$ cd CSC-7333-Skin-Cancer-Classification

# create environment
$ conda create -n cnn-env python=3.10 -y
$ conda activate cnn-env

# install dependencies
$ pip install -r requirements.txt
```

## Folder Structure

```
project_root/
├── data/              # dataset in standard image classification format
├── models/            # trained model parameters
├── data.py            # prepare data
├── main.py            # functions to train&test
├── model.py           # build models
├── train.py           # train PyTorch models
└── utils.py           # utility functions
```

## Run 


```shell
# download dataset
$ python3 data.py

# play with notebooks
$ jupyter notebook

# sample command to train model
$ python3 main.py --model resnet --batch-size 32 --lr 0.001 --num-epochs 10
```
