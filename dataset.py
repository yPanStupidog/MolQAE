import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from rdkit import Chem


class PureMolecularDataset(Dataset):
    """Pure quantum molecular dataset"""
    
    def __init__(self, data_path, max_length=22):
        """
        Initialize dataset
        
        Args:
            data_path: Dataset path (CSV file containing SMILES column)
            max_length: Maximum length of SMILES features
        """
        self.max_length = max_length
        self.regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

        # Read dataset
        data = pd.read_csv(data_path)
        print(f"Loaded '{data_path}' with {len(data)} records.")
        
        # Determine SMILES column name
        smiles_col = "SMILES" if "SMILES" in data.columns else "smiles"
        if smiles_col not in data.columns:
            raise ValueError(f"SMILES column not found, column name should be 'SMILES' or 'smiles'")
            
        # Method to canonicalize the SMILES strings
        def canonicalize(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToSmiles(mol, canonical=True) if mol else None
            except Exception:  # Catch RDKit errors for invalid SMILES
                return None

        # Apply canonicalization to a new temporary column to avoid directly modifying the original 'smiles_col' of 'data' DataFrame
        temp_canonical_col = '_canonical_smiles_temp'  # Temporary column name
        data[temp_canonical_col] = data[smiles_col].apply(canonicalize)
        
        # Remove SMILES that failed canonicalization (these are invalid or cannot be processed by RDKit)
        original_count_before_dropna = len(data)
        data.dropna(subset=[temp_canonical_col], inplace=True)  # Operate on DataFrame 'data'
        print(f"Removed {original_count_before_dropna - len(data)} invalid or non-canonical SMILES.")

        # Remove duplicate canonical SMILES
        original_count_before_dropdup = len(data)
        data.drop_duplicates(subset=[temp_canonical_col], keep='first', inplace=True)
        print(f"Removed {original_count_before_dropdup - len(data)} duplicate canonical SMILES.")

        data.reset_index(drop=True, inplace=True)  # Reset index

        # Preprocess SMILES strings
        self.features_list = []
        smiles_for_features = data[temp_canonical_col].tolist()
        print(f"After preprocessing, {len(smiles_for_features)} valid and unique SMILES are left for feature extraction.")
        print("Start extracting features...")
        for canonical_smiles in smiles_for_features:
            features = self._extract_features(canonical_smiles)  
            self.features_list.append(features)

        del data[temp_canonical_col]

        print(f"--- Dataset initialized, {len(self.features_list)} features extracted ---")

    def _extract_features(self, smiles):
        """Extract features from SMILES string"""
        # Use regex to tokenize
        tokens = re.findall(self.regex_pattern, smiles)
        
        # Create feature vector (using simple one-hot encoding here)
        features = torch.zeros(self.max_length)
        for i, token in enumerate(tokens[:self.max_length]):
            features[i] = sum(ord(c) for c in token) / 255.0  # Normalize to [0,1]
        
        return features
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        return self.features_list[idx]