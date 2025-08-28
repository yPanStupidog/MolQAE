import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np


class PureMolecularQAE(tq.QuantumModule):
    def __init__(self, 
                 n_encoder_qubits=8,
                 n_latent_qubits=4,
                 n_layers=5):
        super().__init__()
        self.n_encoder_qubits = n_encoder_qubits
        self.n_qae_qubits = n_encoder_qubits
        self.n_latent_qubits = n_latent_qubits
        self.n_trash_qubits = n_encoder_qubits - n_latent_qubits  
        self.n_layers = n_layers
        
        # Ensure latent qubits are less than encoder qubits
        if self.n_latent_qubits >= self.n_encoder_qubits:
            raise ValueError(f"n_latent_qubits ({self.n_latent_qubits}) must be less than n_encoder_qubits ({self.n_encoder_qubits})")
        
        # Initialize quantum gate components
        self._init_quantum_components()
    
    def _init_quantum_components(self):
        """Initialize all quantum gate components"""
        # QAE encoder - multi-layer structure
        self.encoder_rotations = nn.ModuleList()
        self.encoder_entanglements = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # 1. Parameterized single-qubit rotation layer
            rotation_layer = nn.ModuleList([
                tq.U3(has_params=True, trainable=True) 
                for _ in range(self.n_qae_qubits)
            ])
            self.encoder_rotations.append(rotation_layer)
            
            # 2. Parameterized two-qubit entanglement layer - fully connected topology
            entanglement_layer = nn.ModuleList()
            for i in range(self.n_qae_qubits):
                for j in range(i+1, self.n_qae_qubits):
                    entanglement_layer.append(tq.CRZ(has_params=True, trainable=True))
            self.encoder_entanglements.append(entanglement_layer)
        
        # QAE decoder - multi-layer structure
        self.decoder_rotations = nn.ModuleList()
        self.decoder_entanglements = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # 1. Parameterized single-qubit rotation layer
            rotation_layer = nn.ModuleList([
                tq.U3(has_params=True, trainable=True) 
                for _ in range(self.n_qae_qubits)
            ])
            self.decoder_rotations.append(rotation_layer)
            
            # 2. Parameterized two-qubit entanglement layer - fully connected topology
            entanglement_layer = nn.ModuleList()
            for i in range(self.n_qae_qubits):
                for j in range(i+1, self.n_qae_qubits):
                    entanglement_layer.append(tq.CRZ(has_params=True, trainable=True))
            self.decoder_entanglements.append(entanglement_layer)
        
        # Add special quantum gate layer: tilted plane entanglement
        self.special_crz_gates = nn.ModuleList([
            tq.CRZ(has_params=True, trainable=True)
            for _ in range(self.n_qae_qubits-1)
        ])
        
        # Add latent space mapping layer
        self.latent_mapping = nn.ModuleList([
            tq.RZ(has_params=True, trainable=True)
            for _ in range(self.n_latent_qubits)
        ])
        
        # Add trash qubit compression layer
        self.trash_compression = nn.ModuleList([
            tq.U3(has_params=True, trainable=True)
            for _ in range(self.n_trash_qubits)
        ])
        
        # Measurement observable
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _reset_qdev(self, bsz=1, device='cpu'):
        """Initialize quantum device and set to all-zero state"""
        qdev = tq.QuantumDevice(n_wires=self.n_encoder_qubits, bsz=bsz, device=device)
        return qdev
    
    def _quantum_encode_smiles(self, qdev, smiles_features):
        """
        Directly encode SMILES features into quantum state
        
        Args:
            qdev: Quantum device
            smiles_features: SMILES feature tensor of shape [batch_size, n_features]
        """
        batch_size = smiles_features.shape[0]
        
        # First stage: Apply parameterized U3 gates to each qubit
        for q_idx in range(self.n_encoder_qubits):
            # Calculate three parameters from feature vector (ensure proper range)
            param_idx_1 = (q_idx * 3) % smiles_features.shape[1]
            param_idx_2 = (q_idx * 3 + 1) % smiles_features.shape[1]
            param_idx_3 = (q_idx * 3 + 2) % smiles_features.shape[1]
            
            # Extract parameters and scale to [0, 2π]
            # Original logic
            theta = smiles_features[:, param_idx_1] * np.pi
            phi = smiles_features[:, param_idx_2] * np.pi
            lam = smiles_features[:, param_idx_3] * np.pi
            
            # Prepare gate parameters
            params = torch.stack([theta, phi, lam], dim=1)
            
            # Apply U3 gate using functional API
            tqf.u3(qdev, wires=q_idx, params=params)
        
        # Second stage: Use CNOT gates to create entanglement (ring topology)
        for q_idx in range(self.n_encoder_qubits):
            tqf.cnot(qdev, wires=[q_idx, (q_idx + 1) % self.n_encoder_qubits])
    
    def forward(self, smiles_features):
        """
        Forward pass function
        
        Args:
            smiles_features: Tensor of shape [batch_size, n_features] representing SMILES features
            
        Returns:
            tuple: (reconstruction fidelity, trash qubit zero-state deviation)
        """
        batch_size = smiles_features.shape[0]
        device = smiles_features.device
        
        # Initialize quantum device
        qdev = self._reset_qdev(bsz=batch_size, device=device)
        
        # Use quantum encoder to encode features into quantum state
        self._quantum_encode_smiles(qdev, smiles_features)
        
        # Save initial quantum state for calculating reconstruction fidelity
        initial_state = qdev.states.clone()
        
        # === Apply QAE encoder circuit ===
        for layer_idx in range(len(self.encoder_rotations)):
            rotation_layer = self.encoder_rotations[layer_idx]
            entanglement_layer = self.encoder_entanglements[layer_idx]
            
            # 1. Apply single-qubit rotations
            for q_idx, rot_gate in enumerate(rotation_layer):
                tqf.u3(qdev, wires=q_idx, params=rot_gate.params)
            
            # 2. Apply two-qubit entanglement
            entangle_idx = 0
            for i in range(self.n_qae_qubits):
                for j in range(i+1, self.n_qae_qubits):
                    tqf.crz(qdev, wires=[i, j], params=entanglement_layer[entangle_idx].params)
                    entangle_idx += 1
            
        # === Middle state processing ===
        # Apply latent space mapping layer - only acts on latent qubits
        for idx, rz_gate in enumerate(self.latent_mapping):
            tqf.rz(qdev, wires=idx, params=rz_gate.params)
        
        # Try to compress trash qubits to |0⟩ state - acts on trash qubits
        for idx, u3_gate in enumerate(self.trash_compression):
            tqf.u3(qdev, wires=self.n_latent_qubits + idx, params=u3_gate.params)
            
        # Calculate trash qubit deviation at the theoretically correct time point 
        # (after trash compression, before special CRZ gates and decoder)
        # Save current state for calculating trash qubit zero-state deviation
        compressed_state = qdev.states.clone()
        # Calculate trash qubit zero-state deviation
        trash_deviation = self._calculate_trash_deviation(compressed_state)
        
        # Special tilted plane entanglement gates - create additional entanglement
        for idx, crz_gate in enumerate(self.special_crz_gates):
            tqf.crz(qdev, wires=[idx, (idx + 1) % self.n_qae_qubits], params=crz_gate.params)
        
        # === Apply QAE decoder circuit ===
        for layer_idx in range(len(self.decoder_rotations)):
            rotation_layer = self.decoder_rotations[layer_idx]
            entanglement_layer = self.decoder_entanglements[layer_idx]
            
            # 1. Apply single-qubit rotations
            for q_idx, rot_gate in enumerate(rotation_layer):
                tqf.u3(qdev, wires=q_idx, params=rot_gate.params)
            
            # 2. Apply two-qubit entanglement
            entangle_idx = 0
            for i in range(self.n_qae_qubits):
                for j in range(i+1, self.n_qae_qubits):
                    tqf.crz(qdev, wires=[i, j], params=entanglement_layer[entangle_idx].params)
                    entangle_idx += 1
        
        # Calculate reconstruction fidelity between initial and final states
        fidelity = self._calculate_fidelity(initial_state, qdev.states)
        
        return fidelity, trash_deviation
    
    def get_latent_state(self, smiles_features):
        """
        Get encoded quantum latent state (before decoder)
        
        Args:
            smiles_features: Tensor of shape [batch_size, n_features] representing SMILES features
            
        Returns:
            tuple: (latent_state, latent_state_1d, trash_deviation)
                - latent_state: Original quantum state
                - latent_state_1d: One-dimensional quantum state representation (for analysis)
                - trash_deviation: Zero-state deviation of trash qubits
        """
        batch_size = smiles_features.shape[0]
        device = smiles_features.device
        
        # Initialize quantum device
        qdev = self._reset_qdev(bsz=batch_size, device=device)
        
        # Use quantum encoder to encode features into quantum state
        self._quantum_encode_smiles(qdev, smiles_features)
        
        # === Apply QAE encoder circuit ===
        for layer_idx in range(len(self.encoder_rotations)):
            rotation_layer = self.encoder_rotations[layer_idx]
            entanglement_layer = self.encoder_entanglements[layer_idx]
            
            # 1. Apply single-qubit rotations
            for q_idx, rot_gate in enumerate(rotation_layer):
                tqf.u3(qdev, wires=q_idx, params=rot_gate.params)
            
            # 2. Apply two-qubit entanglement
            entangle_idx = 0
            for i in range(self.n_qae_qubits):
                for j in range(i+1, self.n_qae_qubits):
                    tqf.crz(qdev, wires=[i, j], params=entanglement_layer[entangle_idx].params)
                    entangle_idx += 1
        
        # === Middle state processing ===
        # Apply latent space mapping layer - only acts on latent qubits
        for idx, rz_gate in enumerate(self.latent_mapping):
            tqf.rz(qdev, wires=idx, params=rz_gate.params)
        
        # Try to compress trash qubits to |0⟩ state - acts on trash qubits
        for idx, u3_gate in enumerate(self.trash_compression):
            tqf.u3(qdev, wires=self.n_latent_qubits + idx, params=u3_gate.params)
        
        # Save compressed state (theoretically correct latent state)
        latent_state = qdev.states.clone()
        
        # Calculate one-dimensional representation of quantum state
        latent_state_1d = self._extract_latent_features(latent_state, batch_size)
        
        # Calculate trash qubit zero-state deviation - using compressed quantum state
        trash_deviation = self._calculate_trash_deviation(latent_state)
        
        return latent_state, latent_state_1d, trash_deviation
    
    def _extract_latent_features(self, quantum_state, batch_size):
        """
        Extract latent feature vectors from quantum state
        
        Args:
            quantum_state: Quantum state tensor
            batch_size: Batch size
            
        Returns:
            torch.Tensor: Extracted feature vector, shape [batch_size, 2^n_qae_qubits]
        """
        # Reshape quantum state to one-dimensional vector
        state_1d = quantum_state.reshape(batch_size, -1)
        
        # Extract amplitude and phase information from quantum state
        amplitudes = torch.abs(state_1d)
        phases = torch.angle(state_1d)
        
        # Combine amplitudes and phases as feature vector
        features = torch.cat([amplitudes, phases], dim=1)
        
        return features
    
    def _calculate_fidelity(self, state1, state2):
        """Calculate fidelity between two quantum states"""
        # First flatten quantum states to 1D vectors
        state1_1d = state1.reshape(state1.shape[0], -1)
        state2_1d = state2.reshape(state2.shape[0], -1)
        
        # Calculate fidelity over batch dimension
        batch_fidelity = torch.abs(torch.sum(torch.conj(state1_1d) * state2_1d, dim=1)) ** 2
        return batch_fidelity.mean()
    
    def _calculate_trash_deviation(self, mid_state):
        """
        Calculate the degree to which trash qubits deviate from the all-zero state
        
        Strictly following quantum autoencoder theory, calculate the probability of trash qubits being in non-zero states

        Args:
            mid_state: Quantum state to analyze (usually the output state of the encoder)
            
        Returns:
            torch.Tensor: Zero-state deviation of trash qubits, scalar value
        """
        # Reshape quantum state to form suitable for probability calculation
        batch_size = mid_state.shape[0]
        states_1d = mid_state.reshape(batch_size, -1)
        
        # Calculate probability distribution of each basis state
        probs = torch.abs(states_1d) ** 2
        
        # Calculate indices corresponding to trash qubits being 0
        # In quantum autoencoders, trash qubits should be compressed to |0⟩ state
        valid_indices = []
        for i in range(2 ** self.n_latent_qubits):
            valid_indices.append(i * (2 ** self.n_trash_qubits))
        
        # Calculate total probability of valid states (trash qubits being 0)
        valid_probs = torch.zeros(batch_size, device=mid_state.device)
        for idx in valid_indices:
            valid_probs += probs[:, idx]
        
        # Calculate probability of trash qubits not being in 0 state
        trash_deviation = 1.0 - valid_probs
        
        return trash_deviation.mean()