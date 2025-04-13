import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def generate_classification_report(true_labels, predictions, class_names, output_path=None):
    """
    Generate and save a classification report
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        output_path: Path to save the report
    
    Returns:
        str: Classification report
    """
    report = classification_report(true_labels, predictions, 
                                  target_names=class_names, 
                                  digits=4)
    
    print(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def plot_confusion_matrix(true_labels, predictions, class_names, output_path=None, figsize=(10, 8)):
    """
    Plot and save a confusion matrix
    
    Args:
        true_labels: True labels
        predictions: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, 
               yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    fig = plt.gcf()
    plt.show()
    
    return fig


def plot_roc_curves(true_labels, probabilities, class_names, output_path=None, figsize=(10, 8)):
    """
    Plot and save ROC curves
    
    Args:
        true_labels: True labels
        probabilities: Predicted probabilities
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert true labels to one-hot encoding
    n_classes = len(class_names)
    true_one_hot = np.zeros((len(true_labels), n_classes))
    for i, val in enumerate(true_labels):
        true_one_hot[i, val] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_one_hot[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=figsize)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    fig = plt.gcf()
    plt.show()
    
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                          output_path=None, figsize=(12, 5)):
    """
    Plot and save training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        output_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not train_losses or not val_losses:
        print("No training history available")
        return None
    
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    fig = plt.gcf()
    plt.show()
    
    return fig