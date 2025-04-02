import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import random

def get_device():
    """
    Utility function to get the available device (CPU or GPU).

    Returns:
        torch.device: The device to use for computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            transform=transforms.Compose([
                transforms.Resize((64, 64)), 
                # transforms.TrivialAugmentWide(num_magnitude_bins=31), # Uncomment if you want to add trivial augmentations
                transforms.ToTensor()
            ])
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
