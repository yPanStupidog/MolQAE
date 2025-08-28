import torch
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
import torchquantum.functional as tqf


def set_random_seed(seed):
    """
    Set all random seeds to ensure experimental reproducibility
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():  # PyTorch GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
        # Set CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Molecular Quantum Autoencoder Training Parameters')
    
    # Add random seed parameter
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Quantum bits related parameters
    parser.add_argument('--n_encoder_qubits', type=int, default=8,
                        help='Number of encoder qubits (default: 8)')
    parser.add_argument('--n_latent_qubits', type=int, default=4,
                        help='Number of latent space qubits (default: 4)')
    parser.add_argument('--n_layers', type=int, default=5,
                        help='Number of quantum circuit layers (default: 5)')
    parser.add_argument('--smiles_feature_dim', type=int, default=22,
                        help='Dimension of SMILES feature vector (default: 22)')
    
    # Training related parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (default: cuda)')
    
    # Data and saving related parameters
    parser.add_argument('--data_path', type=str, default='data/qm9.csv',
                        help='Dataset path (default: data/qm9.csv)')
    parser.add_argument('--save_dir', type=str, default='results/molqae_pure/layer10',
                        help='Base results save directory (default: results/molqae_pure)')
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                        help='Checkpoint saving interval (default: 20)')
    
    args = parser.parse_args()
    
    # Validate that n_latent_qubits is less than n_encoder_qubits
    if args.n_latent_qubits >= args.n_encoder_qubits:
        raise ValueError(f"n_latent_qubits ({args.n_latent_qubits}) must be less than n_encoder_qubits ({args.n_encoder_qubits})")
    
    # Build save path containing encoder and latent qubit information
    args.save_dir = os.path.join(args.save_dir, f"enc{args.n_encoder_qubits}_lat{args.n_latent_qubits}")
    
    return args


def evaluate_pure_model(model, test_dataloader, device='cuda'):
    """
    Evaluate pure quantum model performance
    
    Args:
        model: Trained PureMolecularQAE model
        test_dataloader: Test data loader
        device: Evaluation device
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    total_fidelity = 0.0
    total_trash_deviation = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            features = batch.to(device)
            
            # Forward pass
            fidelity, trash_deviation = model(features)
            
            total_fidelity += fidelity.item()
            total_trash_deviation += trash_deviation.item()
    
    # Calculate average metrics
    avg_fidelity = total_fidelity / len(test_dataloader)
    avg_trash_deviation = total_trash_deviation / len(test_dataloader)
    
    print(f"Evaluation Results:")
    print(f"  Fidelity: {avg_fidelity:.4f}")
    print(f"  Trash Deviation: {avg_trash_deviation:.4f}")
    
    return {
        'fidelity': avg_fidelity,
        'trash_deviation': avg_trash_deviation
    }


def test_trash_deviation_calculation(model, test_dataloader, device='cuda', n_samples=5):
    """
    Test the correctness of trash qubit zero-state deviation calculation method
    
    Compare trash qubit deviations measured at different stages:
    1. After encoder and trash compression (theoretical optimal point)
    2. After complete autoencoder circuit
    
    Args:
        model: Trained PureMolecularQAE model
        test_dataloader: Test data loader
        device: Computing device
        n_samples: Number of samples for testing
        
    Returns:
        dict: Dictionary containing test results
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch of data samples
    batch = next(iter(test_dataloader))
    features = batch[:n_samples].to(device)
    
    # Save comparison results
    results = {
        'post_compression_deviation': [],   # Trash qubit deviation after encoder+compression (theoretical optimal point)
        'post_decoder_deviation': [],       # Trash qubit deviation after decoder
        'fidelity': []                      # Reconstruction fidelity
    }
    
    with torch.no_grad():
        # Process each sample
        for i in range(n_samples):
            feature = features[i:i+1]  # Get single sample
            
            # 1. Initialize quantum device
            qdev = model._reset_qdev(bsz=1, device=device)
            
            # 2. Feature encoding
            model._quantum_encode_smiles(qdev, feature)
            initial_state = qdev.states.clone()
            
            # 3. Apply encoder
            # Apply QAE encoder circuit
            for layer_idx in range(len(model.encoder_rotations)):
                rotation_layer = model.encoder_rotations[layer_idx]
                entanglement_layer = model.encoder_entanglements[layer_idx]
                
                # Apply single-qubit rotations
                for q_idx, rot_gate in enumerate(rotation_layer):
                    tqf.u3(qdev, wires=q_idx, params=rot_gate.params)
                
                # Apply two-qubit entanglement
                entangle_idx = 0
                for i_q in range(model.n_qae_qubits):
                    for j_q in range(i_q+1, model.n_qae_qubits):
                        tqf.crz(qdev, wires=[i_q, j_q], params=entanglement_layer[entangle_idx].params)
                        entangle_idx += 1
            
            # Apply latent space mapping layer
            for idx, rz_gate in enumerate(model.latent_mapping):
                tqf.rz(qdev, wires=idx, params=rz_gate.params)
            
            # Apply trash qubit compression layer
            for idx, u3_gate in enumerate(model.trash_compression):
                tqf.u3(qdev, wires=model.n_latent_qubits + idx, params=u3_gate.params)
            
            # Calculate trash qubit deviation at theoretical optimal time point - after encoder+compression, before special CRZ gates and decoder
            compressed_state = qdev.states.clone()
            post_compression_deviation = model._calculate_trash_deviation(compressed_state)
            
            # Apply special CRZ gates
            for idx, crz_gate in enumerate(model.special_crz_gates):
                tqf.crz(qdev, wires=[idx, (idx + 1) % model.n_qae_qubits], params=crz_gate.params)
            
            # 4. Apply decoder
            for layer_idx in range(len(model.decoder_rotations)):
                rotation_layer = model.decoder_rotations[layer_idx]
                entanglement_layer = model.decoder_entanglements[layer_idx]
                
                # Apply single-qubit rotations
                for q_idx, rot_gate in enumerate(rotation_layer):
                    tqf.u3(qdev, wires=q_idx, params=rot_gate.params)
                
                # Apply two-qubit entanglement
                entangle_idx = 0
                for i_q in range(model.n_qae_qubits):
                    for j_q in range(i_q+1, model.n_qae_qubits):
                        tqf.crz(qdev, wires=[i_q, j_q], params=entanglement_layer[entangle_idx].params)
                        entangle_idx += 1
            
            # Calculate trash qubit deviation after decoder
            decoder_state = qdev.states.clone()
            post_decoder_deviation = model._calculate_trash_deviation(decoder_state)
            
            # Calculate reconstruction fidelity
            fidelity = model._calculate_fidelity(initial_state, decoder_state)
            
            # Save results
            results['post_compression_deviation'].append(post_compression_deviation.item())
            results['post_decoder_deviation'].append(post_decoder_deviation.item())
            results['fidelity'].append(fidelity.item())
    
    # Calculate averages
    results['avg_post_compression_deviation'] = sum(results['post_compression_deviation']) / n_samples
    results['avg_post_decoder_deviation'] = sum(results['post_decoder_deviation']) / n_samples
    results['avg_fidelity'] = sum(results['fidelity']) / n_samples
    
    # Print comparison results
    print(f"Test results (based on {n_samples} samples):")
    print(f"Trash qubit deviation after encoder+compression (theoretical optimal point): {results['avg_post_compression_deviation']:.4f}")
    print(f"Trash qubit deviation after decoder: {results['avg_post_decoder_deviation']:.4f}")
    print(f"Reconstruction fidelity: {results['avg_fidelity']:.4f}")
    print(f"Trash qubit deviation improvement ratio: {(1 - results['avg_post_decoder_deviation']/results['avg_post_compression_deviation'])*100:.2f}%")
    
    return results