import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import numpy as np

from model import PureMolecularQAE
from dataset import PureMolecularDataset
from utils import set_random_seed, parse_args, evaluate_pure_model, test_trash_deviation_calculation


def train_pure_molecular_qae(model, dataloader, n_epochs=100, lr=1e-3, device='cuda', save_path=None, 
                           plot_metrics=True, checkpoint_interval=10):
    """
    Train pure quantum molecular autoencoder
    
    Args:
        model: PureMolecularQAE model
        dataloader: Data loader containing SMILES features
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        save_path: Model save path
        plot_metrics: Whether to plot metrics
        checkpoint_interval: How many epochs to save checkpoint
    """
    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU training")
        device = 'cpu'
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Create save directory
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Training metrics
    metrics = {
        'epochs': []  # Will store per-epoch dictionaries containing all metrics
    }
    
    # Gradient clipping parameter
    grad_clip_val = 1.0
    
    # For early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_fidelity = 0.0
        total_trash_deviation = 0.0
        
        # Use tqdm to display progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch in progress_bar:
            features = batch.to(device)
            
            optimizer.zero_grad()
            
            fidelity, trash_deviation = model(features)
            
            # Calculate loss: aim to maximize fidelity and minimize trash qubit deviation
            fidelity_loss = 1.0 - fidelity  # Minimize 1-fidelity
            
            # Adjust trash qubit deviation penalty weight
            loss = fidelity_loss + 0.01 * trash_deviation
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_fidelity += fidelity.item()
            total_trash_deviation += trash_deviation.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'fidelity': fidelity.item(), 
                'trash': trash_deviation.item()
            })
            
        scheduler.step()
        
        # Calculate average metrics
        avg_loss = total_loss / len(dataloader)
        avg_fidelity = total_fidelity / len(dataloader)
        avg_trash_deviation = total_trash_deviation / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Store metrics in epoch dictionary
        epoch_metrics = {
            'epoch_number': epoch + 1,
            'metrics': {
                'loss': avg_loss,
                'fidelity': avg_fidelity,
                'trash_deviation': avg_trash_deviation,
                'learning_rate': current_lr
            }
        }
        
        # Add epoch metrics to main metrics dictionary
        metrics['epochs'].append(epoch_metrics)
        
        # Save current metrics to JSON file
        if save_path:
            metrics_file = os.path.join(save_path, 'training_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Fidelity: {avg_fidelity:.4f}")
        print(f"  Trash Deviation: {avg_trash_deviation:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            if save_path:
                best_model_path = os.path.join(save_path, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: {patience} epochs without improvement")
                break
        
        # Save checkpoint
        if save_path and (epoch+1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.get_state_dict() if hasattr(scheduler, 'get_state_dict') else scheduler.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final metrics
    if save_path:
        final_model_path = os.path.join(save_path, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # Plot training metrics
    if plot_metrics:
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(1, 3, 1)
        plt.plot([epoch['epoch_number'] for epoch in metrics['epochs']], [epoch['metrics']['loss'] for epoch in metrics['epochs']])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Fidelity
        plt.subplot(1, 3, 2)
        plt.plot([epoch['epoch_number'] for epoch in metrics['epochs']], [epoch['metrics']['fidelity'] for epoch in metrics['epochs']])
        plt.title('Fidelity')
        plt.xlabel('Epoch')
        plt.ylabel('Fidelity')
        
        # Trash deviation
        plt.subplot(1, 3, 3)
        plt.plot([epoch['epoch_number'] for epoch in metrics['epochs']], [epoch['metrics']['trash_deviation'] for epoch in metrics['epochs']])
        plt.title('Trash Deviation')
        plt.xlabel('Epoch')
        plt.ylabel('Deviation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'training_metrics.png'))
        plt.show()
    
    return model, metrics


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU training")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = PureMolecularDataset(args.data_path)

    # Use fixed generator to ensure reproducible dataset splitting
    generator = torch.Generator().manual_seed(args.seed)
    
    # Split into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=generator  # Use fixed generator
    )

    # Create data loaders with fixed workers seed
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=16,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)  # Set different but deterministic seed for each worker
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=16,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)
    )
    
    # Initialize model
    model = PureMolecularQAE(
        n_encoder_qubits=args.n_encoder_qubits,
        n_latent_qubits=args.n_latent_qubits,
        n_layers=args.n_layers
    )
    
    # Print model information
    print(f"Model Configuration:")
    print(f"  Random seed: {args.seed}")
    print(f"  Number of encoder qubits: {args.n_encoder_qubits}")
    print(f"  Number of latent qubits: {args.n_latent_qubits}")
    print(f"  Number of trash qubits: {model.n_trash_qubits}")
    print(f"  Number of quantum circuit layers: {args.n_layers}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")
    
    # Create save directory with experiment-specific path
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")
    
    # Save seed information to a separate file
    seed_info = {
        'random_seed': args.seed,
        'cuda_deterministic': True,
        'cuda_benchmark': False,
        'data_split_seed': args.seed,
        'dataloader_worker_seed_base': args.seed
    }
    with open(os.path.join(args.save_dir, 'random_seed_info.json'), 'w') as f:
        json.dump(seed_info, f, indent=4)
    
    # Train model
    print("Starting model training...")
    model, metrics = train_pure_molecular_qae(
        model=model,
        dataloader=train_dataloader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_dir,
        plot_metrics=True,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Evaluate model
    print("Starting model evaluation...")
    eval_metrics = evaluate_pure_model(model, test_dataloader, device=device)
    
    # Analyze and validate updated trash_deviation calculation method
    print("\nValidating trash qubit zero-state deviation calculation method...")
    trash_analysis = test_trash_deviation_calculation(
        model=model,
        test_dataloader=test_dataloader,
        device=device,
        n_samples=10  # Use 10 samples for testing
    )
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(['After Compression', 'After Decoder'], 
            [trash_analysis['avg_post_compression_deviation'], trash_analysis['avg_post_decoder_deviation']], 
            color=['orange', 'green'])
    plt.title('Trash Qubit Zero-State Deviation Comparison')
    plt.ylabel('Deviation Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.save_dir, 'trash_deviation_comparison.png'))
    
    # Save configuration information
    config_path = os.path.join(args.save_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write('Configuration Parameters:\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
        f.write(f'\nEvaluation Results:\n')
        for metric, value in eval_metrics.items():
            f.write(f'{metric}: {value}\n')
        f.write(f'\nTrash Deviation Analysis:\n')
        f.write(f'Post-Compression Deviation: {trash_analysis["avg_post_compression_deviation"]:.6f}\n')
        f.write(f'Post-Decoder Deviation: {trash_analysis["avg_post_decoder_deviation"]:.6f}\n')
        f.write(f'Improvement Percentage: {(1 - trash_analysis["avg_post_decoder_deviation"]/trash_analysis["avg_post_compression_deviation"])*100:.2f}%\n')
        f.write(f'\nTotal parameters:{total_params:,}\n')
        f.write(f'Total trainable parameters:{trainable_params:,}\n')


if __name__ == "__main__":
    main()