# -*- coding: utf-8 -*-
"""
Script for standard training of MLP, CNN, and Transformer models
on the modular arithmetic task (a + b) mod p.

Saves the trained models for later use in comparative analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F # For loss masking
from torch.optim.lr_scheduler import StepLR

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm.notebook import tqdm # Use tqdm if running in notebook
except ImportError:
    print("tqdm not found. Install with 'pip install tqdm' for progress bars.")
    def tqdm(iterable, **kwargs): # Dummy tqdm if not installed
        return iterable

import random
import math
import os
from typing import Optional, Tuple, List, Dict
from torch import Tensor
import copy

# --- Configuration ---
# General
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_SAVE_DIR = "saved_models_maml" # Keep same directory name
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model Architecture
INPUT_PROJ_DIM = 64
HIDDEN_DIM = 128
N_HEAD = 4
NUM_TF_ENC_LAYERS = 2

# Standard Training (Baseline)
STANDARD_MODULUS = 97
STANDARD_NUM_EPOCHS = 100
STANDARD_BATCH_SIZE = 256
STANDARD_LEARNING_RATE = 1e-3
LR_SCHEDULER_STEP = 10
LR_SCHEDULER_GAMMA = 0.5

# Meta-Learning values needed only for MAX_MODULUS calculation
# If these change in the MAML script, ensure MAX_MODULUS here is >= the MAML one.
_MOCK_META_MODULI_TRAIN = list(range(50, 150))
_MOCK_META_MODULI_TEST = list(range(150, 200))
MAX_MODULUS = max(max(_MOCK_META_MODULI_TRAIN), max(_MOCK_META_MODULI_TEST), STANDARD_MODULUS) # Max p needed anywhere
print(f"Setting model output size based on MAX_MODULUS = {MAX_MODULUS}")


# For reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Data Generation ---

class ModularAdditionDataset(Dataset):
    """Dataset for the (a + b) mod p task for a specific modulus p."""
    def __init__(self, p, max_p_for_norm):
        self.p = p
        self.max_p_norm = float(max_p_for_norm) # Normalization factor
        all_pairs = []
        all_results = []
        for a in range(p):
            for b in range(p):
                all_pairs.append(torch.tensor([a, b], dtype=torch.long))
                all_results.append(torch.tensor((a + b) % p, dtype=torch.long))

        # Store data grouped by class (result c) for K-shot sampling (Not strictly needed for standard, but keeps class consistent)
        self.data_by_class: Dict[int, List[Tuple[Tensor, Tensor]]] = {c: [] for c in range(p)}
        for inp, outp in zip(all_pairs, all_results):
             self.data_by_class[outp.item()].append((inp, outp))

        self.indices = list(range(len(all_pairs)))

    def __len__(self):
        return len(self.indices) # Use indices length for standard dataset

    def __getitem__(self, idx):
         # Standard __getitem__ retrieves based on index in self.indices
         original_idx = self.indices[idx]
         a = original_idx // self.p
         b = original_idx % self.p
         inp = torch.tensor([a, b], dtype=torch.long)
         outp = torch.tensor((a + b) % self.p, dtype=torch.long)
         return inp, outp

# --- 2. Model Definitions ---

# PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# MLP
class MLP(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, hidden_dim):
        super().__init__()
        self.max_p_out = max_p_out
        self.max_p_norm = float(max_p_out)
        self.input_proj = nn.Linear(2, input_proj_dim * 2)
        self.layers = nn.Sequential(
            nn.Linear(input_proj_dim * 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_p_out)
        )
        # Hooks not needed for standard training script

    def forward(self, x):
        x_norm = x.float() / self.max_p_norm
        projected_input = F.relu(self.input_proj(x_norm))
        output = self.layers(projected_input)
        return output

# CNN1D
class CNN1D(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, hidden_dim, num_filters=64, kernel_size=2):
        super().__init__()
        self.max_p_out = max_p_out
        self.max_p_norm = float(max_p_out)
        self.input_proj_dim = input_proj_dim
        self.input_proj = nn.Linear(1, input_proj_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_proj_dim, num_filters, kernel_size=kernel_size),
            nn.ReLU(), nn.BatchNorm1d(num_filters)
        )
        flattened_size = num_filters * 1
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_p_out)
        )
        # Hooks not needed for standard training script

    def forward(self, x):
        a = x[:, 0].float().unsqueeze(1) / self.max_p_norm
        b = x[:, 1].float().unsqueeze(1) / self.max_p_norm
        proj_a = F.relu(self.input_proj(a))
        proj_b = F.relu(self.input_proj(b))
        projected_seq = torch.stack((proj_a, proj_b), dim=1).transpose(1, 2)
        conv_out = self.conv_layers(projected_seq)
        flattened = torch.flatten(conv_out, 1)
        output = self.fc_layers(flattened)
        return output

# TransformerModel
class TransformerModel(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.max_p_out = max_p_out
        self.input_proj_dim = input_proj_dim
        self.token_emb = nn.Embedding(max_p_out, input_proj_dim)
        self.pos_encoder = PositionalEncoding(input_proj_dim, dropout, max_len=3)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_proj_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_proj_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(input_proj_dim, max_p_out)
        # Hooks not needed for standard training script

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        if x.dtype != torch.long: x = x.long()
        emb_a = self.token_emb(x[:, 0])
        emb_b = self.token_emb(x[:, 1])
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, emb_a.unsqueeze(1), emb_b.unsqueeze(1)), dim=1)
        src = self.pos_encoder(src)
        # Simplified forward without manual attention extraction
        memory = self.transformer_encoder(src)
        cls_output = memory[:, 0, :]
        output = self.output_layer(cls_output)
        return output

# --- 3. Utility Functions ---

# train_model function (with early stopping)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="Model", model_save_dir="saved_models", current_p=None, scheduler=None):
    """ Trains model using standard SGD/Adam. Uses sliced outputs for loss. Includes early stopping. Saves best model. """
    print(f"\n--- Training {model_name} (Standard, p={current_p}) ---")
    model.to(device)

    # Early Stopping Parameters
    early_stopping_patience = 10
    early_stopping_threshold = 1.0 # Accuracy percentage points improvement required
    epochs_no_improve = 0
    best_val_acc = -1.0
    val_acc_at_last_significant_improvement = -1.0

    best_model_path = os.path.join(model_save_dir, f"{model_name}_standard_p{current_p}_best.pth")

    if current_p is None:
        raise ValueError("current_p must be provided for standard training loss calculation.")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for batch_idx, (inputs, labels) in enumerate(train_iterator):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            relevant_outputs_for_loss = outputs[:, :current_p]
            loss = criterion(relevant_outputs_for_loss, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(relevant_outputs_for_loss.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_iterator.set_postfix(loss=loss.item())

        train_dataset_size = len(train_loader.dataset) # Use Subset size
        if train_dataset_size > 0:
            train_loss /= train_dataset_size
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        else: train_loss, train_acc = 0.0, 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for inputs, labels in val_iterator:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                relevant_outputs_for_loss = outputs[:, :current_p]
                loss = criterion(relevant_outputs_for_loss, labels)
                val_loss += loss.item() * inputs.size(0)
                relevant_outputs_for_acc = outputs[:, :current_p]
                _, predicted = torch.max(relevant_outputs_for_acc.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_dataset_size = len(val_loader.dataset) # Use Subset size
        if val_dataset_size > 0:
            val_loss /= val_dataset_size
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        else: val_loss, val_acc = 0.0, 0.0

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if scheduler: scheduler.step()

        # Early Stopping Check
        if val_acc > best_val_acc:
            print(f"  Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model to {best_model_path}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            if val_acc > val_acc_at_last_significant_improvement + early_stopping_threshold:
                print(f"    Improvement exceeds threshold ({early_stopping_threshold:.1f}%). Resetting patience count.")
                val_acc_at_last_significant_improvement = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"    Improvement below threshold. Patience count: {epochs_no_improve}/{early_stopping_patience}")
        else:
            epochs_no_improve += 1
            print(f"  Validation accuracy did not improve ({val_acc:.2f}% vs best {best_val_acc:.2f}%). Patience count: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs. No significant improvement (>{early_stopping_threshold:.1f}%) in validation accuracy for {early_stopping_patience} epochs.")
            break

    print(f"Finished Standard Training {model_name}. Best Val Acc: {best_val_acc:.2f}%")
    # No need to load best model here, just ensure it was saved.
    if not os.path.exists(best_model_path):
        print(f"Warning: Best standard model file was NOT saved at {best_model_path}.")

    # Return the model state as it was at the end of training (or early stopping)
    # The MAML script will load the *best* saved state.
    return model


# --- 4. Main Execution Block ---
if __name__ == "__main__":

    # --- Setup ---
    print("--- Standard Model Training Script ---")
    # Standard Dataloaders (fixed p)
    standard_dataset = ModularAdditionDataset(STANDARD_MODULUS, MAX_MODULUS)
    std_total_size = len(standard_dataset) # Use dataset len directly
    std_test_size = int(0.15 * std_total_size); std_val_size = int(0.15 * std_total_size)
    std_train_size = std_total_size - std_test_size - std_val_size
    std_generator = torch.Generator().manual_seed(SEED)
    # Use indices from 0 to len(standard_dataset)-1 for splitting
    std_train_indices, std_val_indices, std_test_indices = random_split(range(std_total_size), [std_train_size, std_val_size, std_test_size], generator=std_generator)
    std_train_subset = Subset(standard_dataset, std_train_indices); std_val_subset = Subset(standard_dataset, std_val_indices); std_test_subset = Subset(standard_dataset, std_test_indices) # Test subset not used for training but defined for consistency
    std_train_loader = DataLoader(std_train_subset, batch_size=STANDARD_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
    std_val_loader = DataLoader(std_val_subset, batch_size=STANDARD_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
    # Test loader definition (optional, not used in this script but good practice)
    std_test_loader = DataLoader(std_test_subset, batch_size=STANDARD_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
    print(f"Standard Dataset (p={STANDARD_MODULUS}): Train={len(std_train_subset)}, Val={len(std_val_subset)}, Test={len(std_test_subset)}")

    # Instantiate models
    mlp_model_std = MLP(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    cnn_model_std = CNN1D(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    transformer_model_std = TransformerModel(MAX_MODULUS, INPUT_PROJ_DIM, nhead=N_HEAD, num_encoder_layers=NUM_TF_ENC_LAYERS).to(DEVICE)

    # Define criterion and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_mlp_std = optim.Adam(mlp_model_std.parameters(), lr=STANDARD_LEARNING_RATE)
    optimizer_cnn_std = optim.Adam(cnn_model_std.parameters(), lr=STANDARD_LEARNING_RATE)
    optimizer_transformer_std = optim.Adam(transformer_model_std.parameters(), lr=STANDARD_LEARNING_RATE)

    # Add LR Schedulers
    scheduler_mlp_std = StepLR(optimizer_mlp_std, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    scheduler_cnn_std = StepLR(optimizer_cnn_std, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    scheduler_transformer_std = StepLR(optimizer_transformer_std, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)


    # --- Train Standard Models ---
    print("\n--- Starting Standard Model Training ---")
    # These calls will train the models and save the best state internally
    _ = train_model(mlp_model_std, std_train_loader, std_val_loader, criterion, optimizer_mlp_std, STANDARD_NUM_EPOCHS, DEVICE, "MLP_Std", MODEL_SAVE_DIR, current_p=STANDARD_MODULUS, scheduler=scheduler_mlp_std)
    _ = train_model(cnn_model_std, std_train_loader, std_val_loader, criterion, optimizer_cnn_std, STANDARD_NUM_EPOCHS, DEVICE, "CNN_Std", MODEL_SAVE_DIR, current_p=STANDARD_MODULUS, scheduler=scheduler_cnn_std)
    _ = train_model(transformer_model_std, std_train_loader, std_val_loader, criterion, optimizer_transformer_std, STANDARD_NUM_EPOCHS, DEVICE, "Transformer_Std", MODEL_SAVE_DIR, current_p=STANDARD_MODULUS, scheduler=scheduler_transformer_std)
    print("\n--- Standard Model Training Script Finished ---")
    print(f"Trained models saved in: {MODEL_SAVE_DIR}")