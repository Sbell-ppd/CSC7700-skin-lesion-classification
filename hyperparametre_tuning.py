import os
import csv
import torch
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from dataset import create_dataloaders
from experiment import train_and_evaluate

class HyperparameterTuner:
    """
    Class to handle automated hyperparameter tuning for skin lesion classification
    """
    def __init__(self, data_path, image_path, backbone='resnet50', num_workers=4, base_results_dir='hyperparameter_results'):
        """
        Initialize the hyperparameter tuner
        
        Args:
            data_path (str or Path): Path to the data directory
            image_path (str or Path): Path to the image directory
            backbone (str): Backbone architecture to tune ('resnet50', 'efficientnet', 'densenet')
            num_workers (int): Number of workers for data loading
            base_results_dir (str): Base directory to store results
        """
        self.data_path = Path(data_path)
        self.image_path = Path(image_path)
        self.num_workers = num_workers
        self.backbone = backbone
        
        # Create results directory with timestamp and backbone name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(base_results_dir) / f"{backbone}_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV file for logging results
        self.results_file = self.results_dir / "results.csv"
        self.create_results_file()
        
        # Dictionary to store best parameters
        self.best_params = {}
        self.best_accuracy = 0.0
        
    def create_results_file(self):
        """Create CSV file with headers for storing experiment results"""
        with open(self.results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Experiment', 'Learning Rate', 'Batch Size', 'Weight Decay', 
                'Dropout Rate', 'Backbone', 'Train Accuracy', 'Val Accuracy', 
                'Test Accuracy', 'Duration (min)'
            ])
    
    def log_result(self, experiment_name, params, train_acc, val_acc, test_acc, duration):
        """Log experiment results to CSV file"""
        with open(self.results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                experiment_name,
                params['learning_rate'],
                params['batch_size'],
                params['weight_decay'],
                params['dropout_rate'],
                params['backbone'],
                f"{train_acc:.4f}",
                f"{val_acc:.4f}",
                f"{test_acc:.4f}",
                f"{duration:.2f}"
            ])
        
        # Update best parameters if this experiment has better validation accuracy
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            self.best_params = params.copy()
            print(f"New best parameters found: {self.best_params}")
    
    def run_experiment(self, experiment_name, params):
        """Run a single experiment with given parameters"""
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"Parameters: {params}")
        print(f"{'='*50}\n")
        
        # Create experiment directory
        experiment_dir = self.results_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        # Create dataloaders
        train_loader, val_loader, test_loader, class_weights = create_dataloaders(
            data_path=self.data_path,
            image_path=self.image_path,
            batch_size=params['batch_size'],
            num_workers=self.num_workers
        )
        
        # Record start time
        start_time = datetime.now()
        
        # Train and evaluate model
        model, trainer, test_accuracy = train_and_evaluate(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            backbone=params['backbone'],
            lr=params['learning_rate'],
            epochs=params['epochs'],
            weight_decay=params['weight_decay'],
            dropout_rate=params['dropout_rate'],
            experiment_name=experiment_name
        )
        
        # Calculate duration in minutes
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        # Get final train and validation accuracy from trainer
        train_accuracy = trainer.train_accs[-1] if trainer.train_accs else 0
        val_accuracy = trainer.val_accs[-1] if trainer.val_accs else 0
        
        # Log results
        self.log_result(
            experiment_name=experiment_name,
            params=params,
            train_acc=train_accuracy,
            val_acc=val_accuracy,
            test_acc=test_accuracy,
            duration=duration
        )
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy
        }
    
    def run_learning_rate_experiments(self):
        """Run experiments to find optimal learning rate"""
        print("\nRunning learning rate experiments...")
        
        learning_rates = [1e-3, 1e-4, 1e-5]
        best_lr = None
        best_val_acc = 0
        
        for i, lr in enumerate(learning_rates):
            params = {
                'learning_rate': lr,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'dropout_rate': 0.5,
                'backbone': self.backbone,
                'epochs': 20  # Reduced epochs for faster exploration
            }
            
            results = self.run_experiment(f"lr_exp_{i+1}", params)
            
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_lr = lr
        
        print(f"\nBest learning rate: {best_lr} (Validation accuracy: {best_val_acc:.4f})")
        return best_lr
    
    def run_batch_size_experiments(self, best_lr):
        """Run experiments to find optimal batch size"""
        print("\nRunning batch size experiments...")
        
        batch_sizes = [16, 32, 64]
        best_bs = None
        best_val_acc = 0
        
        for i, bs in enumerate(batch_sizes):
            params = {
                'learning_rate': best_lr,
                'batch_size': bs,
                'weight_decay': 1e-4,
                'dropout_rate': 0.5,
                'backbone': self.backbone,
                'epochs': 20
            }
            
            results = self.run_experiment(f"bs_exp_{i+1}", params)
            
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_bs = bs
        
        print(f"\nBest batch size: {best_bs} (Validation accuracy: {best_val_acc:.4f})")
        return best_bs
    
    def run_weight_decay_experiments(self, best_lr, best_bs):
        """Run experiments to find optimal weight decay"""
        print("\nRunning weight decay experiments...")
        
        weight_decays = [1e-3, 1e-4, 1e-5]
        best_wd = None
        best_val_acc = 0
        
        for i, wd in enumerate(weight_decays):
            params = {
                'learning_rate': best_lr,
                'batch_size': best_bs,
                'weight_decay': wd,
                'dropout_rate': 0.5,
                'backbone': self.backbone,
                'epochs': 20
            }
            
            results = self.run_experiment(f"wd_exp_{i+1}", params)
            
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_wd = wd
        
        print(f"\nBest weight decay: {best_wd} (Validation accuracy: {best_val_acc:.4f})")
        return best_wd
    
    def run_dropout_experiments(self, best_lr, best_bs, best_wd):
        """Run experiments to find optimal dropout rate"""
        print("\nRunning dropout rate experiments...")
        
        dropout_rates = [0.3, 0.5, 0.7]
        best_dr = None
        best_val_acc = 0
        
        for i, dr in enumerate(dropout_rates):
            params = {
                'learning_rate': best_lr,
                'batch_size': best_bs,
                'weight_decay': best_wd,
                'dropout_rate': dr,
                'backbone': self.backbone,
                'epochs': 20
            }
            
            results = self.run_experiment(f"dr_exp_{i+1}", params)
            
            if results['val_accuracy'] > best_val_acc:
                best_val_acc = results['val_accuracy']
                best_dr = dr
        
        print(f"\nBest dropout rate: {best_dr} (Validation accuracy: {best_val_acc:.4f})")
        return best_dr
    
    def run_final_experiment(self, best_lr, best_bs, best_wd, best_dr):
        """Run final experiment with best parameters"""
        print("\nRunning final experiment with best parameters...")
        
        params = {
            'learning_rate': best_lr,
            'batch_size': best_bs,
            'weight_decay': best_wd,
            'dropout_rate': best_dr,
            'backbone': self.backbone,
            'epochs': 25  # Use more epochs for final training
        }
        
        results = self.run_experiment("final_best_params", params)
        
        print(f"\nFinal results with best parameters:")
        print(f"Train accuracy: {results['train_accuracy']:.4f}")
        print(f"Validation accuracy: {results['val_accuracy']:.4f}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def run_all_experiments(self):
        """Run all hyperparameter tuning experiments in sequence"""
        print("\nStarting hyperparameter tuning process...")
        
        # Phase 1: Find best learning rate
        best_lr = self.run_learning_rate_experiments()
        
        # Phase 2: Find best batch size
        best_bs = self.run_batch_size_experiments(best_lr)
        
        # Phase 3: Find best weight decay
        best_wd = self.run_weight_decay_experiments(best_lr, best_bs)
        
        # Phase 4: Find best dropout rate
        best_dr = self.run_dropout_experiments(best_lr, best_bs, best_wd)
        
        # Phase 5: Final experiment with best parameters
        final_results = self.run_final_experiment(best_lr, best_bs, best_wd, best_dr)
        
        # Generate summary report
        self.generate_summary_report(best_lr, best_bs, best_wd, best_dr, final_results)
        
        return {
            'learning_rate': best_lr,
            'batch_size': best_bs,
            'weight_decay': best_wd,
            'dropout_rate': best_dr,
            'final_results': final_results
        }
    
    def generate_summary_report(self, best_lr, best_bs, best_wd, best_dr, final_results):
        """Generate summary report of all experiments"""
        # Read all results
        df = pd.read_csv(self.results_file)
        
        # Create summary file
        summary_file = self.results_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("=============================\n\n")
            
            f.write("Best Parameters:\n")
            f.write(f"Learning Rate: {best_lr}\n")
            f.write(f"Batch Size: {best_bs}\n")
            f.write(f"Weight Decay: {best_wd}\n")
            f.write(f"Dropout Rate: {best_dr}\n\n")
            
            f.write("Final Results:\n")
            f.write(f"Train Accuracy: {final_results['train_accuracy']:.4f}\n")
            f.write(f"Validation Accuracy: {final_results['val_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {final_results['test_accuracy']:.4f}\n\n")
            
            f.write("Learning Rate Experiments:\n")
            lr_exps = df[df['Experiment'].str.contains('lr_exp')]
            f.write(f"{lr_exps[['Experiment', 'Learning Rate', 'Val Accuracy']].to_string(index=False)}\n\n")
            
            f.write("Batch Size Experiments:\n")
            bs_exps = df[df['Experiment'].str.contains('bs_exp')]
            f.write(f"{bs_exps[['Experiment', 'Batch Size', 'Val Accuracy']].to_string(index=False)}\n\n")
            
            f.write("Weight Decay Experiments:\n")
            wd_exps = df[df['Experiment'].str.contains('wd_exp')]
            f.write(f"{wd_exps[['Experiment', 'Weight Decay', 'Val Accuracy']].to_string(index=False)}\n\n")
            
            f.write("Dropout Rate Experiments:\n")
            dr_exps = df[df['Experiment'].str.contains('dr_exp')]
            f.write(f"{dr_exps[['Experiment', 'Dropout Rate', 'Val Accuracy']].to_string(index=False)}\n\n")
        
        print(f"Summary report generated at {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for skin lesion classification')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image directory')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'efficientnet', 'densenet'],
                        help='Backbone architecture to tune')
    
    # Optional parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--results_dir', type=str, default='hyperparameter_results', 
                        help='Directory to store results')
    
    args = parser.parse_args()
    
    # Initialize and run the hyperparameter tuner
    tuner = HyperparameterTuner(
        data_path=args.data_path,
        image_path=args.image_path,
        backbone=args.backbone,
        num_workers=args.num_workers,
        base_results_dir=args.results_dir
    )
    
    best_params = tuner.run_all_experiments()
    print(f"Hyperparameter tuning completed for {args.backbone}. Best parameters: {best_params}")

if __name__ == "__main__":
    main()
