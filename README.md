# MolQAE: Quantum Autoencoder for Molecular Representation Learning

[![arXiv](https://img.shields.io/badge/arXiv-2505.01875-b31b1b.svg)](https://arxiv.org/abs/2505.01875)
[![IEEE Conference](https://img.shields.io/badge/IEEE%20QAI-2025-blue.svg)](https://qai2025.unina.it/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MolQAE** is a novel quantum autoencoder architecture designed for molecular representation learning. This implementation provides a fully quantum approach to encode molecular SMILES strings into quantum latent spaces for enhanced molecular property prediction and drug discovery applications.

## Reference
This work has been accepted at the **2025 IEEE International Conference on Quantum Artificial Intelligence**. 

If you find this repo useful, please consider citing:
```bibtex
@misc{pan2025molqaequantumautoencodermolecular,
      title={MolQAE: Quantum Autoencoder for Molecular Representation Learning}, 
      author={Yi Pan and Hanqi Jiang and Wei Ruan and Dajiang Zhu and Xiang Li and Yohannes Abate and Yingfeng Wang and Tianming Liu},
      year={2025},
      eprint={2505.01875},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2505.01875}, 
}
```

## Key Features

- **Pure Quantum Architecture**: Implements a fully quantum autoencoder without classical components
- **Molecular Feature Encoding**: Advanced encoding of SMILES molecular features into quantum states
- **Latent Space Compression**: Compresses molecular representations into optimized quantum latent spaces
- **Trash Qubit Optimization**: Novel optimization strategy for trash qubits to maintain zero states during compression
- **Reconstruction Fidelity**: High-precision quantum state reconstruction quality measurement
- **Configurable Architecture**: Supports customizable quantum circuit depths and qubit configurations
- **Reproducible Results**: Comprehensive seed management for experimental reproducibility

## Quick Start

### Prerequisites

- Python 3.12+


### Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n molqae python=3.12 && activate molqae
   ```

2. **Clone MolQAE repository:**
   ```bash
   git clone https://github.com/yPanStupidog/MolQAE.git
   cd MolQAE
   ```

3. **Install TorchQuantum:**
   ```bash
   git clone https://github.com/mit-han-lab/torchquantum.git
   cd torchquantum && pip install --editable . && cd ..
   ```

4. **Install additional dependencies:**
   ```bash
   pip install pandas rdkit torch tqdm matplotlib
   ```

### Dataset Preparation

The project uses the QM9 molecular dataset. Ensure your dataset is placed in the `data/` directory:

```
MolQAE/
└── data/
    └── qm9.csv  # Should contain a 'SMILES' or 'smiles' column
```

## Usage

### Basic Training

Run training with default parameters:

```bash
python train.py
```

### Advanced Configuration

Customize training parameters for your specific requirements:

```bash
python train.py \
    --n_encoder_qubits 8 \
    --n_latent_qubits 4 \
    --n_layers 5 \
    --batch_size 1024 \
    --n_epochs 100 \
    --lr 3e-4 \
    --device cuda \
    --data_path data/qm9.csv \
    --save_dir results/molqae_experiment \
    --seed 42
```

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--n_encoder_qubits` | Number of encoder qubits | 8 | 2-16 |
| `--n_latent_qubits` | Number of latent space qubits | 4 | 1 to `n_encoder_qubits-1` |
| `--n_layers` | Number of quantum circuit layers | 5 | 1-20 |
| `--smiles_feature_dim` | SMILES feature vector dimension | 22 | 10-50 |
| `--batch_size` | Training batch size | 1024 | 32-2048 |
| `--n_epochs` | Number of training epochs | 100 | 10-1000 |
| `--lr` | Learning rate | 3e-4 | 1e-5 to 1e-2 |
| `--device` | Training device | cuda | cuda/cpu |
| `--data_path` | Path to dataset | data/qm9.csv | Valid file path |
| `--save_dir` | Results save directory | results/molqae_pure/layer10 | Valid directory |
| `--checkpoint_interval` | Checkpoint saving interval | 20 | 5-100 |
| `--seed` | Random seed for reproducibility | 42 | Any integer |

## Model Architecture

The MolQAE model consists of several key components:

### 1. Quantum Encoder
- Multi-layer quantum circuit with parameterized single-qubit rotations (U3 gates)
- Fully connected two-qubit entanglement layers (CRZ gates)
- Feature encoding from SMILES molecular strings

### 2. Latent Space Mapping
- RZ gates applied specifically to latent qubits
- Optimized quantum state compression

### 3. Trash Qubit Compression
- U3 gates designed to compress trash qubits to |0⟩ states
- Novel optimization strategy for quantum autoencoder efficiency

### 4. Special Entanglement Layer
- Additional CRZ gates creating tilted plane entanglement
- Enhanced quantum state expressivity

### 5. Quantum Decoder
- Mirror architecture of the encoder for quantum state reconstruction
- High-fidelity molecular representation recovery

## Training Output

The training process generates:

1. **Model Checkpoints**: Saved every N epochs for training resumption
2. **Best Model**: Automatically saved based on lowest loss
3. **Training Metrics**: Real-time loss, fidelity, and trash deviation tracking
4. **Visualization Plots**: Comprehensive training progress visualization
5. **Configuration Files**: Complete parameter logging for reproducibility

### Example Training Output Structure:
```
results/
└── molqae_experiment/
    ├── best_model.pt                 # Best performing model
    ├── final_model.pt               # Final epoch model
    ├── checkpoint_epoch_20.pt       # Periodic checkpoints
    ├── training_metrics.json        # Detailed metrics
    ├── training_metrics.png         # Training plots
    ├── config.txt                   # Configuration summary
    ├── random_seed_info.json        # Reproducibility info
    └── trash_deviation_comparison.png
```

## Model Evaluation

### Programmatic Evaluation

```python
from model import PureMolecularQAE
from dataset import PureMolecularDataset
from utils import evaluate_pure_model
import torch

# Load trained model
model = PureMolecularQAE(n_encoder_qubits=8, n_latent_qubits=4, n_layers=5)
model.load_state_dict(torch.load('path/to/best_model.pt'))

# Prepare test data
test_dataset = PureMolecularDataset('data/qm9.csv', train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

# Evaluate performance
results = evaluate_pure_model(model, test_dataloader, device='cuda')
print(f"Test Fidelity: {results['fidelity']:.4f}")
print(f"Test Trash Deviation: {results['trash_deviation']:.4f}")
```



## Project Structure

```
MolQAE/
├── README.md              # This documentation
├── __init__.py           # Package initialization
├── train.py              # Main training script
├── model.py              # PureMolecularQAE implementation
├── dataset.py            # Dataset loading and preprocessing
├── utils.py              # Utility functions and evaluation
├── data/
│   └── qm9.csv          # QM9 molecular dataset
└── results/             # Training outputs and models
    └── molqae_*/        # Experiment-specific results
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

