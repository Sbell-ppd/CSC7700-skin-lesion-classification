# CSC-7333-Skin-Cancer-Classification

## Project Structure

```
project_root/
├── checkpoints/                        # Saved progress of our training and learned model parameters
├── data/                               # Downloaded image dataset
├── results/                            # Results are saved here
├── dataset.py                          # Dataset loading/transformation into tensors
├── data_preprocessing.ipynb            # Notebok for processing data
├── download_data.py                    # Script for downloading data
├── experiment.py                       # Complete experiment pipeline
├── exploratory-data-analysis.ipynb     # Notebook for exploring data
├── main.py                             # Project entry point
├── model.py                            # Model architecture definitionz
├── trainer.py                          # Training and evaluation logic
├── utils.py                            # Utility functions
└── visualization.py                    # Visualization utilities
```

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

# download dataset
$ python3 download_data.py

# play with notebooks (optional)
$ jupyter notebook

```


## Usage 


```shell
# training and evaluation
$ python main.py --data_path /path/to/data --image_path /path/to/images --backbone resnet50 --epochs 25 --batch_size 32 --experiment_name ham10000_exp1

$ python main.py --data_path="data" --image_path="data/ISIC2018_Task3_Training_Input" --backbone resnet50 --experiment_name experiment1

# visualization Only
$ python main.py --data_path /path/to/data --image_path /path/to/images --visualize --stats
```


### Command-line Arguments

- `--data_path`: Path to the data directory containing metadata
- `--image_path`: Path to the directory containing images
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 25)
- `--lr`: Learning rate (default: 0.0001)
- `--backbone`: Model backbone architecture (default: resnet50)
- `--num_workers`: Number of data loading workers (default: 4)
- `--experiment_name`: Name for the experiment (default: experiment1)
- `--visualize`: Flag to visualize dataset samples
- `--stats`: Flag to display dataset statistics
