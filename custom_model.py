import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVGG(nn.Module):
    """
    TinyVGG architecture for skin lesion classification.
    
    A simplified VGG-style network with fewer parameters, suitable for smaller datasets.
    The architecture consists of multiple blocks of (Conv -> ReLU -> Conv -> ReLU -> MaxPool)
    followed by fully connected layers.
    """
    def __init__(self, num_classes=7, input_channels=3, dropout_rate=0.5):
        super(TinyVGG, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Block 1: Input -> 64 features
        # Input: 224x224x3 -> Output: 112x112x64
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 64 -> 128 features
        # Input: 112x112x64 -> Output: 56x56x128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 128 -> 256 features
        # Input: 56x56x128 -> Output: 28x28x256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: 256 -> 512 features
        # Input: 28x28x256 -> Output: 14x14x512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.block1(x)
        
        # Block 2
        x = self.block2(x)
        
        # Block 3
        x = self.block3(x)
        
        # Block 4
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class TinyVGGWithAttention(nn.Module):
    """
    TinyVGG architecture with attention mechanism for skin lesion classification.
    
    This variant includes a spatial attention mechanism after each block to help
    the model focus on relevant regions of the image.
    """
    def __init__(self, num_classes=7, input_channels=3, dropout_rate=0.5):
        super(TinyVGGWithAttention, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Block 1: Input -> 64 features
        self.block1_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.block1_attention = SpatialAttention()
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 64 -> 128 features
        self.block2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False)
        )
        self.block2_attention = SpatialAttention()
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 128 -> 256 features
        self.block3_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )
        self.block3_attention = SpatialAttention()
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 256 -> 512 features
        self.block4_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )
        self.block4_attention = SpatialAttention()
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.block1_conv(x)
        x = self.block1_attention(x) * x  # Apply attention
        x = self.block1_pool(x)
        
        # Block 2
        x = self.block2_conv(x)
        x = self.block2_attention(x) * x  # Apply attention
        x = self.block2_pool(x)
        
        # Block 3
        x = self.block3_conv(x)
        x = self.block3_attention(x) * x  # Apply attention
        x = self.block3_pool(x)
        
        # Block 4
        x = self.block4_conv(x)
        x = self.block4_attention(x) * x  # Apply attention
        x = self.block4_pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for focusing on relevant image regions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        # Spatial attention uses max and avg pooling along channel dimension,
        # then combines them with a convolution to get an attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Get max and mean values along channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        
        # Conv and sigmoid to get attention weights
        attention = self.conv(pooled)
        attention = self.sigmoid(attention)
        
        return attention


def create_custom_model(model_type='standard', num_classes=7, dropout_rate=0.5):
    """
    Factory function to create a TinyVGG model
    
    Args:
        model_type: 'standard' or 'attention' to choose model architecture
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        A PyTorch model
    """
    if model_type == 'standard':
        return TinyVGG(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'attention':
        return TinyVGGWithAttention(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard' or 'attention'.")


if __name__ == "__main__":
    # Test the model
    model = create_custom_model('standard', num_classes=7)
    print(model)
    
    # Test with a sample input
    sample_input = torch.randn(2, 3, 224, 224)  # batch_size, channels, height, width
    output = model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [2, 7]
    
    # Calculate number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Test attention model
    attention_model = create_custom_model('attention', num_classes=7)
    attention_output = attention_model(sample_input)
    print(f"Attention model output shape: {attention_output.shape}")  # Should be [2, 7]
    print(f"Attention model parameters: {count_parameters(attention_model):,}")