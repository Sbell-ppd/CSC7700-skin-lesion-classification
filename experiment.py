import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dataset import create_dataloaders, display_dataset_stats
from custom_model import create_custom_cnn
from visualization import plot_training_curves

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


def train_custom_model(data_path, image_path, model_type='standard', 
                      initial_filters=32, dropout_rate=0.3, 
                      batch_size=32, num_epochs=50, learning_rate=0.001, 
                      experiment_name='custom_cnn_experiment'):
    """
    Train and evaluate a custom CNN model
    
    Args:
        data_path: Path to the data directory
        image_path: Path to the image directory
        model_type: 'standard' or 'residual'
        initial_filters: Number of filters in the first conv layer
        dropout_rate: Dropout rate for regularization
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        experiment_name: Name for this experiment
    
    Returns:
        tuple: (model, accuracy, history)
    """
    data_path = Path(data_path)
    image_path = Path(image_path)
    
    # Make sure directories exist
    checkpoints_dir = Path(f"checkpoints/{experiment_name}")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(f"results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_path=data_path,
        image_path=image_path,
        batch_size=batch_size
    )
    
    # Display dataset statistics
    print("Dataset statistics:")
    display_dataset_stats(train_loader.dataset, val_loader.dataset, test_loader.dataset)
    
    # Create model
    print(f"Creating custom CNN model (type: {model_type})...")
    model = create_custom_cnn(
        model_type=model_type, 
        num_classes=7, 
        initial_filters=initial_filters, 
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Define learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Initialize tracking variables
    best_val_acc = 0.0
    best_model_weights = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stop_counter = 0
    early_stop_patience = 10
    
    # Train the model
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} - {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Track metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                # Update learning rate based on validation loss
                scheduler.step(epoch_loss)
                
                # Deep copy the model if it's the best validation accuracy so far
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_weights = model.state_dict().copy()
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, checkpoints_dir / "best_model.pth")
                    
                    print(f"Saved new best model with validation accuracy: {epoch_acc:.4f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': epoch_acc,
            }, checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth")
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs without improvement")
            break
        
        print()
    
    # Plot training curves
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        output_path=results_dir / "training_curves.png"
    )
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    # Evaluate on test set
    model.eval()
    test_corrects = 0
    test_losses = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_losses += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_loss = test_losses / len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    
    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    
    # Generate classification report
    class_names = [
        'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
        'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion'
    ]
    
    generate_classification_report(
        np.array(all_labels), 
        np.array(all_preds), 
        class_names,
        output_path=results_dir / "classification_report.txt"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(all_labels), 
        np.array(all_preds), 
        class_names,
        output_path=results_dir / "confusion_matrix.png"
    )
    
    # Plot ROC curves
    plot_roc_curves(
        np.array(all_labels), 
        np.array(all_probs), 
        class_names,
        output_path=results_dir / "roc_curves.png"
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'initial_filters': initial_filters,
        'dropout_rate': dropout_rate,
        'test_acc': test_acc.item()
    }, results_dir / "final_model.pth")
    
    return model, test_acc.item(), history


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