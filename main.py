import argparse
from pathlib import Path
import torch

from dataset import create_dataloaders, visualize_dataset_samples, visualize_batch, display_dataset_stats
from experiment import train_and_evaluate, train_custom_model, predict_image


def main():
    """
    Main function to run the training and evaluation process
    """
    parser = argparse.ArgumentParser(description='Skin Lesion Classification')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image directory')

    # Model parameters (for custom CNN)
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'residual'],
                       help='Type of custom CNN architecture')
    parser.add_argument('--initial_filters', type=int, default=32, 
                       help='Number of filters in first conv layer')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet', 'densenet'], help='Backbone architecture')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--experiment_name', type=str, default='experiment1', help='Name for the experiment')
    parser.add_argument('--visualize', action='store_true', help='Visualize dataset samples')
    parser.add_argument('--stats', action='store_true', help='Display dataset statistics')
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    data_path = Path(args.data_path)
    image_path = Path(args.image_path)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_path=data_path,
        image_path=image_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Visualize dataset samples if requested
    if args.visualize:
        print("Visualizing dataset samples...")
        visualize_dataset_samples(train_loader.dataset, num_samples=10)
        print("Visualizing a batch from the training loader...")
        visualize_batch(train_loader)
    
    # Display dataset statistics if requested
    if args.stats:
        print("Displaying dataset statistics...")
        display_dataset_stats(
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset
        )
    
    # Train and evaluate model
    if args.model_type == 'standard' or args.model_type == 'residual':
        print("Training custom CNN model...")
        model, accuracy, history = train_custom_model(
            data_path=args.data_path,
            image_path=args.image_path,
            model_type=args.model_type,
            initial_filters=args.initial_filters,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            experiment_name=args.experiment_name
        )
    else:
        model, trainer, accuracy = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            backbone=args.backbone,
            lr=args.lr,
            epochs=args.epochs,
            experiment_name=args.experiment_name
        )
    
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model and results saved to 'results/{args.experiment_name}'")


if __name__ == "__main__":
    main()