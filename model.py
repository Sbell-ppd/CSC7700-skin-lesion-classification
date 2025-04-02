import torch
import torch.nn as nn
import torchvision.models as models

class SkinLesionModel(nn.Module):
    """
    CNN model for skin lesion classification based on a pre-trained backbone
    """
    def __init__(self, num_classes=7, backbone='resnet50', pretrained=True, dropout_rate=0.5):
        super(SkinLesionModel, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained model as backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove avg pool and FC
            self.feature_dim = 2048
            
        elif backbone == 'efficientnet':
            base_model = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = base_model.features
            self.feature_dim = base_model.classifier[1].in_features
            
        elif backbone == 'densenet':
            base_model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_extractor = base_model.features
            self.feature_dim = base_model.classifier.in_features
            
        else:
            raise ValueError(f"Backbone '{backbone}' not supported. Choose from 'resnet50', 'efficientnet', 'densenet'.")
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Global pooling
        pooled = self.global_pool(features).view(x.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


def create_model(num_classes=7, backbone='resnet50', pretrained=True, dropout_rate=0.5):
    """
    Factory function to create a new model instance
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone architecture ('resnet50', 'efficientnet', 'densenet')
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        
    Returns:
        SkinLesionModel: Initialized model
    """
    return SkinLesionModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )