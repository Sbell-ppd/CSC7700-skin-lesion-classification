import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from utils import get_device

class SkinDataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion images.
    """
    def __init__(self, metadata_df, image_dir, transform=None):
        """
        Args:
            metadata_df (pandas.DataFrame): DataFrame containing image metadata
            image_dir (str or Path): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.metadata_df = metadata_df
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Create a mapping of diagnosis to numerical labels
        self.diagnosis_mapping = {
            'akiec': 0,  # Actinic Keratoses and Intraepithelial Carcinoma
            'bcc': 1,    # Basal Cell Carcinoma
            'bkl': 2,    # Benign Keratosis-like Lesions
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic Nevi
            'vasc': 6    # Vascular Lesions
        }
        
        # Filter out images that don't exist
        valid_rows = []
        for _, row in self.metadata_df.iterrows():
            image_id = row['image_id']
            if self._find_image_path(image_id) is not None:
                valid_rows.append(row)
        
        # Create a new DataFrame with only valid images
        if valid_rows:
            self.metadata_df = pd.DataFrame(valid_rows)
        else:
            print("Warning: No valid images found!")
            # Create an empty DataFrame with the same columns
            self.metadata_df = pd.DataFrame(columns=self.metadata_df.columns)

    def _find_image_path(self, image_id):
        """Find the full path to an image based on its ID"""
        # Check common extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        """Get an item by index"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get metadata for this image
        row = self.metadata_df.iloc[idx]
        image_id = row['image_id']
        
        # Get image path
        img_path = self._find_image_path(image_id)
        if img_path is None:
            raise FileNotFoundError(f"Image file for {image_id} not found")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label (diagnosis)
        label = self.diagnosis_mapping[row['dx']]
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataloaders(data_path, image_path, batch_size=32, test_size=0.2, val_size=0.1, random_state=42, num_workers=4):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        data_path (str or Path): Path to the data directory
        image_path (str or Path): Path to the directory containing images
        batch_size (int): Batch size for DataLoaders
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    data_path = Path(data_path)
    image_path = Path(image_path)
    
    # Load metadata
    metadata_file = data_path / "HAM10000_metadata.csv"
    if not metadata_file.exists():
        # Try looking for the metadata file in the parent directory
        metadata_file = data_path.parent / "HAM10000_metadata.csv"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    
    metadata_df = pd.read_csv(metadata_file)
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalization values for pre-trained models
        # Note: Normalization values are standard for pre-trained models like ResNet
        # These values are commonly used for models trained on ImageNet
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 70% training, 10% validation, 20% testing 
    # Split data into train and test
    train_df, test_df = train_test_split(
        metadata_df, test_size=test_size, random_state=random_state, stratify=metadata_df['dx']
    )
    
    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_df['dx']
    )
    
    # Create datasets
    train_dataset = SkinDataset(train_df, image_path, transform=train_transform)
    val_dataset = SkinDataset(val_df, image_path, transform=val_test_transform)
    test_dataset = SkinDataset(test_df, image_path, transform=val_test_transform)

    print(f"Created datasets for train, validation, and test sets.")
    
    # Calculate class weights for handling class imbalance
    label_counts = train_df['dx'].value_counts()
    total_samples = len(train_df)
    class_weights = []
    
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }

    print("Label counts in training set:")
    for dx, count in label_counts.items():
        print(f"  {dx}: {count}")
    
    # Calculate weights
    for dx in sorted(label_mapping, key=label_mapping.get):
        weight = total_samples / (len(label_mapping) * label_counts[dx])
        class_weights.append(weight)
    
    device = get_device()
    # Normalize class weights to prevent extreme values - particularly in classification tasks, 
    # class weights are used to handle imbalanced datasets. If some classes have significantly 
    # more samples than others, the model may become biased toward the majority classes. Class weights 
    # assign higher importance to underrepresented classes during training, helping the model learn more balanced decision boundaries.
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Created dataloaders with:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    print(f"  Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights


def visualize_dataset_samples(dataset, num_samples=10, rows=2, cols=5, figsize=(15, 8), 
                              with_transforms=False, random_seed=None):
    """
    Visualize random samples from a PyTorch dataset
    
    Args:
        dataset: PyTorch Dataset object
        num_samples: Number of samples to display
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size (width, height) in inches
        with_transforms: If True, will use the dataset's transforms. If False, will show original images
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Create a temporary dataset with no transforms if with_transforms is False
    if not with_transforms:
        temp_dataset = type(dataset)(
            dataset.metadata_df, 
            dataset.image_dir, 
            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        )
        vis_dataset = temp_dataset
    else:
        vis_dataset = dataset
    
    # Adjust number of samples if needed
    num_samples = min(num_samples, len(vis_dataset))
    
    # Get the diagnosis mapping (reverse it for display)
    reverse_diagnosis_mapping = {v: k for k, v in dataset.diagnosis_mapping.items()}
    
    # Map diagnosis codes to full names for better readability
    diagnosis_fullnames = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesion'
    }
    
    # Generate random indices
    indices = random.sample(range(len(vis_dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    # Function to denormalize images if needed
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Display each image
    for i, idx in enumerate(indices):
        if i >= len(axes):  # Ensure we don't exceed the grid size
            break
            
        # Get image and label
        image, label = vis_dataset[idx]
        
        # Denormalize if the image is a tensor and has values below 1.0
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                if image.min() < 0:  # Check if image is normalized
                    image = denormalize(image)
            # Convert tensor to numpy for display
            image = image.permute(1, 2, 0).numpy()
            # Clip values to valid range
            image = np.clip(image, 0, 1)
        
        # Get the diagnosis code
        diagnosis_code = reverse_diagnosis_mapping[label]
        # Get the full diagnosis name
        diagnosis_name = diagnosis_fullnames.get(diagnosis_code, diagnosis_code)
        
        # Display the image
        axes[i].imshow(image)
        axes[i].set_title(f"{diagnosis_name} ({diagnosis_code})", fontsize=10)
        axes[i].axis('off')
    
    # Remove any unused subplots
    for i in range(num_samples, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualize_batch(dataloader, batch_idx=0, max_images=20, figsize=(15, 8), denormalize=True):
    """
    Visualize a batch of images from a DataLoader
    
    Args:
        dataloader: PyTorch DataLoader
        batch_idx: Which batch to visualize
        max_images: Maximum number of images to display
        figsize: Figure size (width, height) in inches
        denormalize: Whether to denormalize images (assuming ImageNet normalization)
    """
    # Get the first batch
    dataiter = iter(dataloader)
    for _ in range(batch_idx + 1):
        try:
            images, labels = next(dataiter)
        except StopIteration:
            print(f"Error: Dataloader only has {_} batches, requested batch {batch_idx}")
            return None
    
    # Get the dataset from the dataloader
    dataset = dataloader.dataset
    
    # Get the diagnosis mapping (reverse it for display)
    reverse_diagnosis_mapping = {v: k for k, v in dataset.diagnosis_mapping.items()}
    
    # Map diagnosis codes to full names for better readability
    diagnosis_fullnames = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesion'
    }
    
    # Determine grid dimensions
    num_images = min(len(images), max_images)
    cols = 5
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Function to denormalize images
    def denorm(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # Display each image
    for i in range(num_images):
        # Get image and label
        image = images[i].cpu()
        label = labels[i].item()
        
        # Denormalize if requested
        if denormalize:
            image = denorm(image)
        
        # Convert to numpy
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        
        # Get the diagnosis code
        diagnosis_code = reverse_diagnosis_mapping[label]
        # Get the full diagnosis name
        diagnosis_name = diagnosis_fullnames.get(diagnosis_code, diagnosis_code)
        
        # Display the image
        axes[i].imshow(image)
        axes[i].set_title(f"{diagnosis_name} ({diagnosis_code})", fontsize=10)
        axes[i].axis('off')
    
    # Remove any unused subplots
    for i in range(num_images, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return fig


def display_dataset_stats(train_dataset, val_dataset=None, test_dataset=None):
    """
    Display statistics about the datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
    """
    datasets = {
        'Training': train_dataset,
        'Validation': val_dataset,
        'Testing': test_dataset
    }
    
    # Map diagnosis codes to full names for better readability
    diagnosis_fullnames = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesion'
    }
    
    print("Dataset Statistics:")
    print("=" * 50)
    
    # Get the reverse mapping from numerical to string labels
    reverse_mapping = {v: k for k, v in train_dataset.diagnosis_mapping.items()}

    # Dictionary to store counts for plotting
    all_counts = {}
    
    for name, dataset in datasets.items():
        if dataset is None:
            continue
            
        print(f"\n{name} Set:")
        print(f"  Total samples: {len(dataset)}")
        
        # Count labels
        label_counts = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label_str = reverse_mapping[label]
            label_counts[label_str] = label_counts.get(label_str, 0) + 1
        
        # Store counts for this dataset
        all_counts[name] = label_counts

        # Print label distribution
        print(f"  Class distribution:")
        for label, count in sorted(label_counts.items()):
            percentage = (count / len(dataset)) * 100
            full_name = diagnosis_fullnames.get(label, label)
            print(f"    - {full_name} ({label}): {count} samples ({percentage:.2f}%)")
    
    print("\n" + "=" * 50)

    # Create visualizations of the class distribution
    if train_dataset is not None:
        plt.figure(figsize=(14, 6))
        
        # Get all unique labels across all datasets
        all_labels = sorted(set().union(*[counts.keys() for counts in all_counts.values()]))
        
        # Prepare data for plotting
        x = np.arange(len(all_labels))
        width = 0.8 / len(all_counts)  # Width of bars
        
        # Plot bars for each dataset
        for i, (name, counts) in enumerate(all_counts.items()):
            # Get counts for each label (0 if not present)
            dataset_counts = [counts.get(label, 0) for label in all_labels]
            plt.bar(x + i*width - width*len(all_counts)/2 + width/2, dataset_counts, 
                    width=width, label=name)
        
        # Add labels and title
        plt.xlabel('Diagnosis')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution Across Datasets')
        plt.xticks(x, [f"{diagnosis_fullnames.get(label, label)}\n({label})" for label in all_labels], 
                  rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Create pie chart for training dataset
        plt.figure(figsize=(10, 10))
        train_counts = all_counts['Training']
        labels = [f"{diagnosis_fullnames.get(label, label)} ({label}): {count}" 
                 for label, count in sorted(train_counts.items())]
        counts = [count for _, count in sorted(train_counts.items())]
        
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.tab10.colors, shadow=True)
        plt.axis('equal')
        plt.title('Training Set Class Distribution')
        plt.tight_layout()
        plt.show()
    
    return all_counts

if __name__ == "__main__":
    data_path = Path("data/")
    image_path = data_path / "ISIC2018_Task3_Training_Input"
    
    # turn dataset into dataloaders, processes images in batches
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_path, image_path, batch_size=32
    )
    
    # Show a sample batch
    dataiter = iter(train_loader)
    image, label = next(dataiter)
    print(f"Batch shape: {image.shape}") # [batch_size, color_channels, height, width]
    print(f"Labels: {label}") # Display the labels for the first batch

    # Use class weights with loss function for adjusting the loss calculation to account for class imbalance
    # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # visualize_dataset_samples(train_loader.dataset, num_samples=10)




