import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

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
        
        # Verify images exist
        self.valid_image_ids = []
        for idx, row in self.metadata_df.iterrows():
            image_id = row['image_id']
            if self._find_image_path(image_id) is not None:
                self.valid_image_ids.append(idx)
        
        # Filter metadata to only include images that exist
        self.metadata_df = self.metadata_df.iloc[self.valid_image_ids]

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

def create_dataloaders(data_path, image_path, batch_size=32, test_size=0.2, val_size=0.1, random_state=42):
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
    
    # Calculate class weights for handling class imbalance
    label_counts = train_df['dx'].value_counts()
    total_samples = len(train_df)
    class_weights = []
    
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }
    
    # Calculate weights
    for dx in sorted(label_mapping, key=label_mapping.get):
        weight = total_samples / (len(label_mapping) * label_counts[dx])
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )
    
    print(f"Created dataloaders with:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    print(f"  Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    data_path = Path("data/")
    image_path = data_path / "ISIC2018_Task3_Training_Input"
    
    # turn dataset into dataloaders, processes images in batches
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_path, image_path, batch_size=32
    )
    
    # Show a sample batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # Use class weights with loss function for handling imbalance
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
