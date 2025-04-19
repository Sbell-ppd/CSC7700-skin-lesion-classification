# CSC-7333-Skin-Cancer-Classification

## Project Structure

```
project_root/
├── checkpoints/                        # Saved progress of our training and learned model parameters
├── custom_cnn.py                       # Custom models architecture
├── data/                               # Downloaded image dataset
├── results/                            # Results are saved here
├── dataset.py                          # Dataset loading/transformation into tensors
├── data_preprocessing.ipynb            # Notebok for processing data
├── download_data.py                    # Script for downloading data
├── experiment.py                       # Complete experiment pipeline
├── exploratory-data-analysis.ipynb     # Notebook for exploring data
├── main.py                             # Project entry point
├── model.py                            # Model architecture based on resnet/densenet/efficientnet
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

### Custom models

```shell
# Custom models - standard, residual
$ python main.py --data_path /path/to/data --image_path /path/to/images --model_type residual --epochs 50

# example usage
$ python main.py --data_path="data" --image_path="data/ISIC2018_Task3_Training_Input" --model_type standard
```

#### Command-line Arguments
- `--data_path`: Path to the data directory containing metadata
- `--image_path`: Path to the directory containing images
- `--model_type`: Choose 'standard' or 'residual'
- `--initial_filters`: Number of filters in first layer (default: 32)
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 25)
- `--lr`: Learning rate (default: 0.0001)
- `--experiment_name`: Name for the experiment (default: experiment1)


### Backbone models

```shell
# Backbone models - RNN, EfficientNet, DenseNet
# training and evaluation
$ python main.py --data_path /path/to/data --image_path /path/to/images --backbone resnet50 --epochs 25 --batch_size 32 --experiment_name ham10000_exp1

# example usage
$ python main.py --data_path="data" --image_path="data/ISIC2018_Task3_Training_Input" --backbone resnet50 --experiment_name experiment1

# visualization Only
$ python main.py --data_path /path/to/data --image_path /path/to/images --visualize --stats
```


#### Command-line Arguments

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


## Using Individual Components

### Dataset and Data Loading

```python
from dataset import create_dataloaders, visualize_dataset_samples

train_loader, val_loader, test_loader, class_weights = create_dataloaders(
    data_path="data/",
    image_path="data/ISIC2018_Task3_Training_Input",
    batch_size=32
)

# Visualize samples
visualize_dataset_samples(train_loader.dataset, num_samples=10)
```

### Model Creation

```python
from model import create_model

model = create_model(num_classes=7, backbone='resnet50', pretrained=True)
```

### Training

```python
from trainer import LesionClassifier
import torch.nn as nn
from torch.optim import Adam

trainer = LesionClassifier(model, device='cuda')
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=0.0001)

trainer.train_model(
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    num_epochs=25
)
```

### Evaluation

```python
accuracy, loss, predictions, true_labels, probabilities = trainer.evaluate(test_loader)
```

### Visualization

```python
from visualization import plot_confusion_matrix, plot_roc_curves

plot_confusion_matrix(true_labels, predictions, class_names)
plot_roc_curves(true_labels, probabilities, class_names)
```

## Results

After training, results will be saved in the `results/{experiment_name}` directory:

- `classification_report.txt`: Precision, recall, and F1-score for each class
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for each class
- `final_model.pth`: Trained model weights

