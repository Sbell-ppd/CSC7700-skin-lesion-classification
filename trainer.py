import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from tqdm import tqdm
import time
import copy
import os
from pathlib import Path
from utils import get_device


class LesionClassifier:
    """
    Wrapper class for training, evaluating and using the model
    """
    def __init__(self, model, class_names=None):
        """
        Initialize the classifier with a model and device
        
        Args:
            model: PyTorch model
            device: Device to use (cuda or cpu)
            class_names: List of class names
        """
        self.model = model
        self.device = get_device()
        self.model.to(self.device)
        
        # Default class names if not provided
        if class_names is None:
            self.class_names = [
                'Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
                'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion'
            ]
        else:
            self.class_names = class_names
        
        # Class to diagnosis code mapping
        self.idx_to_diagnosis = {
            0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
        }
        
        self.best_model_wts = None
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs=25, 
                    scheduler=None, early_stopping=5, experiment_name='experiment'):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of epochs
            scheduler: Learning rate scheduler (optional)
            early_stopping: Number of epochs to wait before early stopping
            experiment_name: Name for saving model checkpoints
        
        Returns:
            model: Trained model
        """
        print(f"Training on {self.device}...")
        
        # Create directory for checkpoints if it doesn't exist
        checkpoints_dir = Path(f"checkpoints/{experiment_name}")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        early_stop_counter = 0
        
        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    self.model.eval()   # Set model to evaluate mode
                    dataloader = val_loader
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1} - {phase}"):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels) # calculate loss
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward() # backpropagation
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                # Update learning rate if scheduler provided
                if phase == 'train' and scheduler is not None:
                    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Track metrics
                if phase == 'train':
                    self.train_losses.append(epoch_loss)
                    self.train_accs.append(epoch_acc.item())
                else:
                    self.val_losses.append(epoch_loss)
                    self.val_accs.append(epoch_acc.item())
                
                # Deep copy the model if best validation accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, checkpoints_dir / f"best_model.pth")
                    
                    print(f"Saved new best model with validation accuracy: {epoch_acc:.4f}")
                    early_stop_counter = 0
                elif phase == 'val':
                    early_stop_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth")
            
            # Early stopping
            if early_stop_counter >= early_stopping:
                print(f"Early stopping triggered after {early_stopping} epochs without improvement")
                break
                
            print()
        
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        self.best_model_wts = best_model_wts
        self.best_acc = best_acc
        
        # Plot training curves
        self.plot_training_curves(checkpoints_dir / "training_curves.png")
        
        return self.model
    
    def evaluate(self, test_loader, criterion=None):
        """
        Evaluate the model on the test set
        
        Args:
            test_loader: DataLoader for test set
            criterion: Loss function (optional)
        
        Returns:
            tuple: (accuracy, loss, predictions, true_labels)
        """
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)
                
                # Track loss if criterion provided
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = running_corrects.double() / len(test_loader.dataset)
        
        if criterion is not None:
            loss = running_loss / len(test_loader.dataset)
            print(f'Test Loss: {loss:.4f} Acc: {accuracy:.4f}')
        else:
            loss = None
            print(f'Test Acc: {accuracy:.4f}')
        
        return accuracy, loss, np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def predict(self, image):
        """
        Predict the class of a single image
        
        Args:
            image: PyTorch tensor [1, C, H, W]
        
        Returns:
            tuple: (predicted_class_idx, probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item(), probs.cpu().numpy()[0]
    
    def predict_batch(self, images):
        """
        Predict the classes of a batch of images
        
        Args:
            images: PyTorch tensor [B, C, H, W]
        
        Returns:
            tuple: (predicted_class_indices, probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy(), probs.cpu().numpy()
    
    def save_model(self, path):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'idx_to_diagnosis': self.idx_to_diagnosis
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path, model, device=None):
        """
        Load a model
        
        Args:
            path: Path to the saved model
            model: Model architecture (unintialized)
            device: Device to load the model to
        
        Returns:
            LesionClassifier: Initialized classifier with loaded model
        """
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        class_names = checkpoint.get('class_names', None)
        
        classifier = cls(model, device, class_names)
        
        if 'idx_to_diagnosis' in checkpoint:
            classifier.idx_to_diagnosis = checkpoint['idx_to_diagnosis']
        
        print(f"Model loaded from {path}")
        return classifier