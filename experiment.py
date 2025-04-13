import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import os
from pathlib import Path

from model import create_model
from trainer import LesionClassifier
from visualization import generate_classification_report, plot_confusion_matrix, plot_roc_curves
from utils import get_device


def train_and_evaluate(train_loader, val_loader, test_loader, class_weights=None, 
                       backbone='resnet50', lr=0.0001, epochs=25, experiment_name='experiment1'):
    """
    Train and evaluate a model
    
    Args:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        class_weights: Tensor of class weights for weighted loss
        backbone: Backbone architecture ('resnet50', 'efficientnet', 'densenet')
        lr: Learning rate
        epochs: Number of epochs
        experiment_name: Name for experiment (used for saving checkpoints)
    
    Returns:
        tuple: (model, trainer, accuracy)
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Build model
    model = create_model(num_classes=7, backbone=backbone)
    
    # Define class names
    class_names = [
        'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
        'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion'
    ]
    
    # Create trainer
    trainer = LesionClassifier(model, class_names)
    
    # Define loss function with class weights if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Define scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Train model
    model = trainer.train_model(
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=epochs, 
        scheduler=scheduler, 
        early_stopping=10,
        experiment_name=experiment_name
    )
    
    # Create results directory
    results_dir = Path(f"results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on test set
    accuracy, loss, predictions, true_labels, probabilities = trainer.evaluate(test_loader, criterion)
    
    # Generate classification report
    generate_classification_report(
        true_labels, 
        predictions, 
        class_names,
        output_path=results_dir / "classification_report.txt"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        true_labels, 
        predictions, 
        class_names,
        output_path=results_dir / "confusion_matrix.png"
    )
    
    # Plot ROC curves
    plot_roc_curves(
        true_labels, 
        probabilities, 
        class_names,
        output_path=results_dir / "roc_curves.png"
    )
    
    # Save model
    trainer.save_model(results_dir / "final_model.pth")
    
    return model, trainer, accuracy


def predict_image(model_path, image_tensor, backbone='resnet50'):
    """
    Predict the class of a single image using a trained model
    
    Args:
        model_path: Path to the trained model file
        image_tensor: A preprocessed image tensor [1, C, H, W]
        backbone: The backbone architecture used for the model
        
    Returns:
        tuple: (predicted_class_name, predicted_class_code, probabilities)
    """
    # Create model with same architecture
    model = create_model(num_classes=7, backbone=backbone)
    
    # Load trained model
    classifier = LesionClassifier.load_model(model_path, model)
    
    # Make prediction
    class_idx, probs = classifier.predict(image_tensor)
    
    # Get class name and code
    reverse_mapping = {v: k for k, v in classifier.idx_to_diagnosis.items()}
    class_code = reverse_mapping[class_idx]
    class_name = classifier.class_names[class_idx]
    
    return class_name, class_code, probs