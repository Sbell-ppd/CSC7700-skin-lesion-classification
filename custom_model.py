import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block with batch normalization and optional dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_batchnorm=True, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        
        layers = []
        # Conv layer
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=not use_batchnorm)
        )
        
        # Batch normalization
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        layers.append(nn.ReLU(inplace=True))
        
        # Dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for skin lesion classification
    """
    def __init__(self, num_classes=7, input_channels=3, initial_filters=32, 
                 dropout_rate=0.3, use_batchnorm=True):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Entry block
        self.entry_block = nn.Sequential(
            ConvBlock(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, 
                      use_batchnorm=use_batchnorm, dropout_rate=dropout_rate/2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # First stage - learning lower level features
        self.stage1 = nn.Sequential(
            ConvBlock(initial_filters, initial_filters, use_batchnorm=use_batchnorm, dropout_rate=0),
            ConvBlock(initial_filters, initial_filters*2, use_batchnorm=use_batchnorm, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second stage - learning mid-level features
        self.stage2 = nn.Sequential(
            ConvBlock(initial_filters*2, initial_filters*2, use_batchnorm=use_batchnorm, dropout_rate=0),
            ConvBlock(initial_filters*2, initial_filters*4, use_batchnorm=use_batchnorm, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third stage - learning higher level features
        self.stage3 = nn.Sequential(
            ConvBlock(initial_filters*4, initial_filters*4, use_batchnorm=use_batchnorm, dropout_rate=0),
            ConvBlock(initial_filters*4, initial_filters*8, use_batchnorm=use_batchnorm, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth stage - learning complex features
        self.stage4 = nn.Sequential(
            ConvBlock(initial_filters*8, initial_filters*8, use_batchnorm=use_batchnorm, dropout_rate=0),
            ConvBlock(initial_filters*8, initial_filters*16, use_batchnorm=use_batchnorm, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(initial_filters*16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
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
        # Feature extraction
        x = self.entry_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class CustomCNNWithResiduals(nn.Module):
    """
    Custom CNN architecture with residual connections for skin lesion classification
    """
    def __init__(self, num_classes=7, input_channels=3, initial_filters=32, 
                 dropout_rate=0.3, use_batchnorm=True):
        super(CustomCNNWithResiduals, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Entry block
        self.conv1 = ConvBlock(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, 
                              use_batchnorm=use_batchnorm, dropout_rate=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # First residual block
        self.res1_conv1 = ConvBlock(initial_filters, initial_filters, use_batchnorm=use_batchnorm, dropout_rate=0)
        self.res1_conv2 = ConvBlock(initial_filters, initial_filters, use_batchnorm=use_batchnorm, dropout_rate=0)
        
        # Second residual block with down-sampling
        self.res2_downsample = nn.Sequential(
            nn.Conv2d(initial_filters, initial_filters*2, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(initial_filters*2)
        )
        self.res2_conv1 = ConvBlock(initial_filters, initial_filters*2, stride=2, 
                                    use_batchnorm=use_batchnorm, dropout_rate=0)
        self.res2_conv2 = ConvBlock(initial_filters*2, initial_filters*2, 
                                    use_batchnorm=use_batchnorm, dropout_rate=dropout_rate)
        
        # Third residual block with down-sampling
        self.res3_downsample = nn.Sequential(
            nn.Conv2d(initial_filters*2, initial_filters*4, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(initial_filters*4)
        )
        self.res3_conv1 = ConvBlock(initial_filters*2, initial_filters*4, stride=2, 
                                    use_batchnorm=use_batchnorm, dropout_rate=0)
        self.res3_conv2 = ConvBlock(initial_filters*4, initial_filters*4, 
                                    use_batchnorm=use_batchnorm, dropout_rate=dropout_rate)
        
        # Fourth residual block with down-sampling
        self.res4_downsample = nn.Sequential(
            nn.Conv2d(initial_filters*4, initial_filters*8, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(initial_filters*8)
        )
        self.res4_conv1 = ConvBlock(initial_filters*4, initial_filters*8, stride=2, 
                                    use_batchnorm=use_batchnorm, dropout_rate=0)
        self.res4_conv2 = ConvBlock(initial_filters*8, initial_filters*8, 
                                    use_batchnorm=use_batchnorm, dropout_rate=dropout_rate)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(initial_filters*8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
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
        # Entry block
        x = self.conv1(x)
        x = self.pool1(x)
        
        # First residual block
        identity = x
        x = self.res1_conv1(x)
        x = self.res1_conv2(x)
        x += identity
        x = F.relu(x)
        
        # Second residual block
        identity = self.res2_downsample(x)
        x = self.res2_conv1(x)
        x = self.res2_conv2(x)
        x += identity
        x = F.relu(x)
        
        # Third residual block
        identity = self.res3_downsample(x)
        x = self.res3_conv1(x)
        x = self.res3_conv2(x)
        x += identity
        x = F.relu(x)
        
        # Fourth residual block
        identity = self.res4_downsample(x)
        x = self.res4_conv1(x)
        x = self.res4_conv2(x)
        x += identity
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_custom_cnn(model_type='standard', num_classes=7, initial_filters=32, 
                     dropout_rate=0.3, use_batchnorm=True):
    """
    Factory function to create a custom CNN model
    
    Args:
        model_type: 'standard' or 'residual' to choose model architecture
        num_classes: Number of output classes
        initial_filters: Number of filters in the first conv layer
        dropout_rate: Dropout rate for regularization
        use_batchnorm: Whether to use batch normalization
        
    Returns:
        A PyTorch model
    """
    if model_type == 'standard':
        return CustomCNN(
            num_classes=num_classes,
            initial_filters=initial_filters,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm
        )
    elif model_type == 'residual':
        return CustomCNNWithResiduals(
            num_classes=num_classes,
            initial_filters=initial_filters,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard' or 'residual'.")


if __name__ == "__main__":
    # Example usage
    
    # Create a standard custom CNN
    cnn_model = create_custom_cnn('standard', num_classes=7, initial_filters=32)
    
    # Print model summary
    print(cnn_model)
    
    # Test with a sample input
    sample_input = torch.randn(2, 3, 224, 224)  # batch_size, channels, height, width
    output = cnn_model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [2, 7]
    
    # Create a custom CNN with residual connections
    res_model = create_custom_cnn('residual', num_classes=7, initial_filters=32)
    
    # Print model summary
    print(res_model)
    
    # Test with a sample input
    output = res_model(sample_input)
    print(f"Output shape: {output.shape}")  # Should be [2, 7]
    
    # Calculate number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Standard CNN parameters: {count_parameters(cnn_model):,}")
    print(f"Residual CNN parameters: {count_parameters(res_model):,}")