# -*- coding: utf-8 -*-
"""
Experiment comparing internal representations of MLP, CNN, and Transformer
on a modular arithmetic task (a + b) mod p, comparing standard training
against Model-Agnostic Meta-Learning (MAML).

Includes:
- Data Generation for standard and meta-learning tasks (varying moduli)
- Model Definitions (MLP, 1D CNN, Transformer Encoder) using direct numerical inputs + projection
- Activation Extraction Utility
- Standard Model Training Loop (Revised Loss Calculation, LR Scheduler)
- MAML Meta-Training Loop (using 'higher' library)
- Interpretability Functions (updated for comparison):
    - Dimensionality Reduction Visualization (PCA/t-SNE)
    - Representational Similarity Analysis (RSA)
    - Probing (Linear Classifiers)
    - Attention Map Visualization (for Transformer)
- Main execution block that trains models (standard & MAML) and runs analysis
  comparing standard, MAML-meta, and MAML-adapted representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F # For loss masking
from torch.optim.lr_scheduler import StepLR # Added LR Scheduler

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm.notebook import tqdm # Use tqdm if running in notebook
except ImportError:
    print("tqdm not found. Install with 'pip install tqdm' for progress bars.")
    def tqdm(iterable, **kwargs): # Dummy tqdm if not installed
        return iterable
try:
    import higher # For MAML gradient handling
except ImportError:
    print("higher library not found. Install with 'pip install higher'. MAML training will fail.")
    higher = None # Set to None if not available

import random
import math
import os # For saving models
from typing import Optional, Tuple, List, Dict # Added for type hints
from torch import Tensor # Added for type hints
import copy # For deep copying models during adaptation analysis

# --- Configuration ---
# General
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_SAVE_DIR = "saved_models_maml"
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Dir should already exist

# Model Architecture
INPUT_PROJ_DIM = 64 # Dimension after initial projection of normalized a, b
HIDDEN_DIM = 128 # General hidden dimension size
N_HEAD = 4 # Transformer heads
NUM_TF_ENC_LAYERS = 2 # Transformer layers

# Standard Training (Baseline)
STANDARD_MODULUS = 97 # Fixed modulus for standard training
STANDARD_NUM_EPOCHS = 100 # Increased epochs
STANDARD_BATCH_SIZE = 256
STANDARD_LEARNING_RATE = 1e-3 # Starting learning rate
LR_SCHEDULER_STEP = 10 # Scheduler step size (epochs)
LR_SCHEDULER_GAMMA = 0.5 # Scheduler decay factor

# Meta-Learning (MAML)
META_MODULI_TRAIN = list(range(50, 150)) # Range of moduli for meta-training tasks
META_MODULI_TEST = list(range(150, 200)) # Range of moduli for meta-testing tasks
# Filter to keep only primes (optional, but common)
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0: return False
    return True
META_MODULI_TRAIN = [p for p in META_MODULI_TRAIN if is_prime(p)]
META_MODULI_TEST = [p for p in META_MODULI_TEST if is_prime(p)]
print(f"Using {len(META_MODULI_TRAIN)} train moduli (primes {min(META_MODULI_TRAIN)}-{max(META_MODULI_TRAIN)})")
print(f"Using {len(META_MODULI_TEST)} test moduli (primes {min(META_MODULI_TEST)}-{max(META_MODULI_TEST)})")

MAX_MODULUS = max(max(META_MODULI_TRAIN), max(META_MODULI_TEST), STANDARD_MODULUS) # Max p needed anywhere
print(f"Maximum modulus considered: {MAX_MODULUS}")

META_BATCH_SIZE = 5 # Number of tasks (moduli) per meta-batch
META_NUM_EPOCHS = 50 # Meta-training epochs (outer loop iterations)
META_LR = 1e-3 # Meta-optimizer learning rate (outer loop)
INNER_LR = 1e-2 # Inner loop learning rate (adaptation step)
NUM_INNER_STEPS = 5 # Number of adaptation steps in inner loop
K_SHOT = 10 # Number of examples per class in support/query sets (N-way K-shot)

# For reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- 1. Data Generation ---

class ModularAdditionDataset(Dataset):
    """Dataset for the (a + b) mod p task for a specific modulus p."""
    def __init__(self, p, max_p_for_norm): # Changed max_p usage
        self.p = p
        self.max_p_norm = float(max_p_for_norm) # Normalization factor
        # Removed self.inputs = []
        # Removed self.outputs = []
        # Generate all pairs
        all_pairs = []
        all_results = []
        for a in range(p):
            for b in range(p):
                # Store integers, normalization happens in model forward pass
                all_pairs.append(torch.tensor([a, b], dtype=torch.long))
                all_results.append(torch.tensor((a + b) % p, dtype=torch.long))

        # Store data grouped by class (result c) for K-shot sampling
        self.data_by_class: Dict[int, List[Tuple[Tensor, Tensor]]] = {c: [] for c in range(p)}
        for inp, outp in zip(all_pairs, all_results):
             self.data_by_class[outp.item()].append((inp, outp))

        self.indices = list(range(len(all_pairs))) # Store original indices if needed

    def __len__(self):
        # For MAML dataset, length is based on grouped data size
        return sum(len(v) for v in self.data_by_class.values())

    def __getitem__(self, idx):
         # Find the item corresponding to the original index idx
         # This might not be directly used if sampling via sample_task_batch
         # but implement for completeness / potential other uses
         count = 0
         for c, items in self.data_by_class.items():
             if idx < count + len(items):
                 item_idx = idx - count
                 return items[item_idx]
             count += len(items)
         raise IndexError("Index out of range in MAML dataset __getitem__")

    def sample_task_batch(self, k_shot, batch_type='support'):
        """Samples a support or query batch (k examples per class)."""
        support_inputs, support_outputs = [], []
        query_inputs, query_outputs = [], []

        for c in range(self.p): # Iterate through all possible results (classes)
            if not self.data_by_class[c]: continue
            if len(self.data_by_class[c]) < 2 * k_shot:
                samples_needed = 2 * k_shot
                available_samples = self.data_by_class[c]
                indices_to_sample = [i % len(available_samples) for i in range(samples_needed)]
                class_samples = [available_samples[i] for i in indices_to_sample]
                random.shuffle(class_samples)
            else:
                class_samples = random.sample(self.data_by_class[c], 2 * k_shot)

            support_samples = class_samples[:k_shot]
            query_samples = class_samples[k_shot:]

            support_inputs.extend([s[0] for s in support_samples])
            support_outputs.extend([s[1] for s in support_samples])
            query_inputs.extend([q[0] for q in query_samples])
            query_outputs.extend([q[1] for q in query_samples])

        if not support_inputs or not query_inputs:
             print(f"Warning: Could not sample sufficient data for task p={self.p}, k={k_shot}. Returning empty batch.")
             return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

        support_inputs_t = torch.stack(support_inputs)
        support_outputs_t = torch.stack(support_outputs)
        query_inputs_t = torch.stack(query_inputs)
        query_outputs_t = torch.stack(query_outputs)

        support_perm = torch.randperm(support_inputs_t.size(0))
        query_perm = torch.randperm(query_inputs_t.size(0))

        # Return integer tensors
        return (support_inputs_t[support_perm], support_outputs_t[support_perm],
                query_inputs_t[query_perm], query_outputs_t[query_perm])

    def forward(self, x):
        x_norm = x.float() / self.max_p_norm
        projected_input = F.relu(self.input_proj(x_norm))
        output = self.layers(projected_input)
        return output

# --- 2. Model Definitions (Revised Input Processing) ---

# Added PositionalEncoding class (needed for Transformer)
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
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 2.1 MLP ---
class MLP(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, hidden_dim): # Takes max_p_out for output layer size
        super().__init__()
        self.max_p_out = max_p_out
        self.max_p_norm = float(max_p_out) # Use output dim for normalization factor
        # Input projection layer for normalized [a, b]
        self.input_proj = nn.Linear(2, input_proj_dim * 2) # Project 2 features to match old embedding dim
        self.layers = nn.Sequential(
            # Input to first linear layer is now input_proj_dim * 2
            nn.Linear(input_proj_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_p_out) # Output layer size = max_p_out
        )
        self.activations = {}
        self._hook_handles = []
        # Hooks registered dynamically

    def _get_activation(self, name):
        def hook(model, input, output):
            # Input is a tuple for sequential, take first element
            act_input = input[0] if isinstance(input, tuple) else input
            self.activations[name + '_input'] = act_input.detach()
            self.activations[name + '_output'] = output.detach()
        return hook

    def _register_hooks(self):
        self.remove_hooks()
        self._hook_handles.append(self.input_proj.register_forward_hook(self._get_activation('input_proj')))
        self._hook_handles.append(self.layers[0].register_forward_hook(self._get_activation('linear_1')))
        self._hook_handles.append(self.layers[3].register_forward_hook(self._get_activation('linear_2')))
        self._hook_handles.append(self.layers[6].register_forward_hook(self._get_activation('pre_softmax')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}

    def forward(self, x):
        # x shape: (batch_size, 2) - containing integers a and b
        # Normalize input
        x_norm = x.float() / self.max_p_norm # Shape: (batch_size, 2)
        # Project input
        projected_input = F.relu(self.input_proj(x_norm)) # Shape: (batch_size, input_proj_dim * 2)
        # Pass through MLP layers
        output = self.layers(projected_input) # Output shape (batch_size, max_p_out)
        return output

# --- 2.2 CNN (1D) ---
class CNN1D(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, hidden_dim, num_filters=64, kernel_size=2):
        super().__init__()
        self.max_p_out = max_p_out
        self.max_p_norm = float(max_p_out)
        self.input_proj_dim = input_proj_dim
        # Project normalized 'a' and 'b' separately
        self.input_proj = nn.Linear(1, input_proj_dim)
        self.conv_layers = nn.Sequential(
            # Input shape: (batch_size, input_proj_dim, seq_len=2)
            nn.Conv1d(input_proj_dim, num_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters)
        )
        flattened_size = num_filters * 1
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_p_out) # Output size max_p_out
        )
        self.activations = {}
        self._hook_handles = []
        # Hooks registered dynamically

    def _get_activation(self, name):
        def hook(model, input, output):
            act_input = input[0] if isinstance(input, tuple) else input
            self.activations[name + '_input'] = act_input.detach()
            self.activations[name + '_output'] = output.detach()
        return hook

    def _register_hooks(self):
        self.remove_hooks()
        self._hook_handles.append(self.input_proj.register_forward_hook(self._get_activation('input_proj')))
        self._hook_handles.append(self.conv_layers[0].register_forward_hook(self._get_activation('conv_1')))
        self._hook_handles.append(self.fc_layers[0].register_forward_hook(self._get_activation('dense_1')))
        self._hook_handles.append(self.fc_layers[3].register_forward_hook(self._get_activation('pre_softmax')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}

    def forward(self, x):
        # x shape: (batch_size, 2) - integers a, b
        a = x[:, 0].float().unsqueeze(1) / self.max_p_norm # Shape (batch_size, 1)
        b = x[:, 1].float().unsqueeze(1) / self.max_p_norm # Shape (batch_size, 1)

        # Project a and b separately
        proj_a = F.relu(self.input_proj(a)) # Shape (batch_size, input_proj_dim)
        proj_b = F.relu(self.input_proj(b)) # Shape (batch_size, input_proj_dim)

        # Stack projections to form a sequence: (batch_size, seq_len=2, input_proj_dim)
        projected_seq = torch.stack((proj_a, proj_b), dim=1)
        # Transpose for Conv1d: (batch_size, input_proj_dim, seq_len=2)
        projected_seq = projected_seq.transpose(1, 2)

        conv_out = self.conv_layers(projected_seq)
        flattened = torch.flatten(conv_out, 1)
        output = self.fc_layers(flattened) # Output shape (batch_size, max_p_out)
        return output

# --- 2.3 Transformer (Encoder Only) ---
class TransformerModel(nn.Module):
    def __init__(self, max_p_out, input_proj_dim, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.max_p_out = max_p_out
        self.max_p_norm = float(max_p_out) # Still needed? Maybe not directly for input, but could be useful context. Keep for now.
        self.input_proj_dim = input_proj_dim # This is effectively d_model
        # Use nn.Embedding for integer inputs 'a' and 'b'
        self.token_emb = nn.Embedding(max_p_out, input_proj_dim) # Embed integers from 0 to max_p_out-1
        # Remove self.input_proj = nn.Linear(1, input_proj_dim)
        self.pos_encoder = PositionalEncoding(input_proj_dim, dropout, max_len=3) # max_len=3 for [CLS, emb_a, emb_b]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_proj_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_proj_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(input_proj_dim, max_p_out) # Output size max_p_out
        self.activations = {}
        self.attention_weights = {}
        self._hook_handles = []
        # Hooks registered dynamically

    def _get_activation(self, name):
        def hook(model, input, output):
             # Handle potential tuple inputs/outputs for layers like TransformerEncoderLayer
            act_input = input[0] if isinstance(input, tuple) else input
            act_output = output[0] if isinstance(output, tuple) else output
            # Detach before storing
            if isinstance(act_input, torch.Tensor): self.activations[name + '_input'] = act_input.detach()
            if isinstance(act_output, torch.Tensor): self.activations[name + '_output'] = act_output.detach()
        return hook

    def _register_hooks(self):
        self.remove_hooks()
        # Remove hook for input_proj
        # self._hook_handles.append(self.input_proj.register_forward_hook(self._get_activation('input_proj')))
        # Optional: Add hook for token_emb if needed
        # self._hook_handles.append(self.token_emb.register_forward_hook(self._get_activation('token_emb')))
        for i, layer in enumerate(self.transformer_encoder.layers):
            self._hook_handles.append(layer.register_forward_hook(self._get_activation(f'encoder_layer_{i}')))
        self._hook_handles.append(self.output_layer.register_forward_hook(self._get_activation('pre_softmax')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, 2) - integers a, b
        batch_size = x.size(0)
        # Remove normalization and projection
        # a = x[:, 0].float().unsqueeze(1) / self.max_p_norm # Shape (batch_size, 1)
        # b = x[:, 1].float().unsqueeze(1) / self.max_p_norm # Shape (batch_size, 1)
        # proj_a = F.relu(self.input_proj(a)) # Shape (batch_size, input_proj_dim)
        # proj_b = F.relu(self.input_proj(b)) # Shape (batch_size, input_proj_dim)

        # Get embeddings for a and b
        # Ensure input tensor is Long type for embedding lookup
        if x.dtype != torch.long:
             x = x.long()
        emb_a = self.token_emb(x[:, 0]) # Shape (batch_size, input_proj_dim)
        emb_b = self.token_emb(x[:, 1]) # Shape (batch_size, input_proj_dim)


        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape (batch_size, 1, input_proj_dim)

        # Combine CLS token with embeddings: (batch_size, 3, input_proj_dim)
        # Unsqueeze emb_a and emb_b to add sequence dimension before concatenating
        src = torch.cat((cls_tokens, emb_a.unsqueeze(1), emb_b.unsqueeze(1)), dim=1)

        # Apply positional encoding
        src = self.pos_encoder(src)

        # --- Encoder Pass & Manual Attention Extraction ---
        self.attention_weights = {}
        current_input = src
        for i, layer in enumerate(self.transformer_encoder.layers):
            memory = layer(current_input)
            # Manual attention extraction (assuming norm_first=False)
            q = k = v = current_input
            _, attn_weights = layer.self_attn(q, k, v, need_weights=True, average_attn_weights=True)
            self.attention_weights[f'encoder_layer_{i}_attention'] = attn_weights.detach()
            current_input = memory
        # --- End Encoder Loop ---

        cls_output = memory[:, 0, :] # Use the output corresponding to the CLS token
        output = self.output_layer(cls_output) # Output shape (batch_size, max_p_out)
        return output

# --- 3. Utility Functions ---

# Custom Loss Function for varying modulus p
def masked_cross_entropy_loss(outputs, targets, p):
    """Calculates cross entropy loss, slicing outputs up to p."""
    relevant_outputs = outputs[:, :p]
    return F.cross_entropy(relevant_outputs, targets)


# Standard Training Function (Revised Loss Calculation + LR Scheduler)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="Model", model_save_dir="saved_models", current_p=None, scheduler=None): # Added scheduler
    """ Trains model using standard SGD/Adam. Uses sliced outputs for loss. Includes early stopping. """
    # This function is no longer used in this script, kept for reference or potential future use.
    # The actual training happens in modular_arithmetic_vanilla.py
    pass # Remove or comment out the body if desired


# MAML Meta-Training Function
def meta_train_model(model, meta_train_moduli, k_shot, num_inner_steps, inner_lr, meta_optimizer, num_epochs, device, model_name="Model", model_save_dir="saved_models", meta_batch_size=5):
    """ Trains model using MAML """
    if higher is None: raise ImportError("The 'higher' library is required for MAML training.")

    print(f"\n--- Training {model_name} (MAML) ---")
    model.to(device)
    best_meta_val_acc = -1.0
    best_model_path = os.path.join(model_save_dir, f"{model_name}_maml_meta_best.pth")
    train_task_datasets = {p: ModularAdditionDataset(p, model.max_p_out) for p in meta_train_moduli} # Use max_p_out
    print(f"Created datasets for {len(train_task_datasets)} meta-training tasks.")

    outer_iterator = tqdm(range(num_epochs), desc="Meta Epochs")
    for meta_epoch in outer_iterator:
        model.train()
        meta_batch_loss = 0.0
        meta_batch_acc = 0.0
        tasks_processed = 0

        if len(meta_train_moduli) < meta_batch_size: task_batch_moduli = meta_train_moduli
        else: task_batch_moduli = random.sample(meta_train_moduli, meta_batch_size)

        meta_optimizer.zero_grad()
        task_iterator = tqdm(task_batch_moduli, desc=f"Meta Epoch {meta_epoch+1} Tasks", leave=False)
        for task_p in task_iterator:
            task_dataset = train_task_datasets[task_p]
            support_x, support_y, query_x, query_y = task_dataset.sample_task_batch(k_shot)
            if support_x.numel() == 0 or query_x.numel() == 0: continue

            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            try:
                with higher.innerloop_ctx(model, meta_optimizer, copy_initial_weights=True, device=device, track_higher_grads=True) as (fmodel, diffopt):
                    for step in range(num_inner_steps):
                        support_outputs = fmodel(support_x)
                        support_loss = masked_cross_entropy_loss(support_outputs, support_y, task_p) # Use masked loss here too
                        diffopt.step(support_loss)

                    query_outputs = fmodel(query_x)
                    query_loss = masked_cross_entropy_loss(query_outputs, query_y, task_p) # Use masked loss

                    effective_batch_size = len(task_batch_moduli)
                    task_meta_loss = query_loss / effective_batch_size
                    task_meta_loss.backward()

                    with torch.no_grad():
                         relevant_outputs = query_outputs[:, :task_p]
                         _, predicted = torch.max(relevant_outputs.data, 1)
                         correct = (predicted == query_y).sum().item()
                         meta_batch_acc += 100 * correct / query_y.size(0)

                    meta_batch_loss += task_meta_loss.item()
                    tasks_processed += 1
            except Exception as e:
                print(f"\nError during inner loop/backward for task p={task_p}: {e}")

        if tasks_processed > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            meta_optimizer.step()
            avg_meta_loss = meta_batch_loss
            avg_meta_acc = meta_batch_acc / tasks_processed
            outer_iterator.set_postfix(meta_loss=f"{avg_meta_loss:.4f}", query_acc=f"{avg_meta_acc:.2f}%")
            if avg_meta_acc > best_meta_val_acc:
                best_meta_val_acc = avg_meta_acc
                torch.save(model.state_dict(), best_model_path)
        else: print(f"Meta Epoch {meta_epoch+1}: No tasks processed successfully.")

    print(f"Finished MAML Training {model_name}. Best Avg Query Acc: {best_meta_val_acc:.2f}%")
    if os.path.exists(best_model_path):
        print(f"Loading best MAML meta model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else: print(f"Warning: Best MAML meta model file not found at {best_model_path}.")
    return model


# Function to perform MAML adaptation for analysis
def adapt_maml_model(meta_model, task_p, k_shot, num_inner_steps, inner_lr, device):
    """Adapts a meta-trained model to a specific task p."""
    print(f"Adapting MAML model to task p={task_p}...")
    meta_model.to(device)
    task_dataset = ModularAdditionDataset(task_p, meta_model.max_p_out) # Use max_p_out
    support_x, support_y, _, _ = task_dataset.sample_task_batch(k_shot)

    if support_x.numel() == 0:
         print(f"Cannot adapt model to p={task_p}: Failed to sample support data.")
         return copy.deepcopy(meta_model)

    support_x, support_y = support_x.to(device), support_y.to(device)
    adapted_model = copy.deepcopy(meta_model)
    adapted_model.train()
    # Use Adam for adaptation as well, might be more stable than SGD
    adapter_optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)

    print(f"  Adaptation Inner Loop (p={task_p}):")
    for step in range(num_inner_steps):
        adapter_optimizer.zero_grad()
        outputs = adapted_model(support_x)
        loss = masked_cross_entropy_loss(outputs, support_y, task_p) # Use masked loss
        loss.backward()
        adapter_optimizer.step()
        print(f"    Step {step+1}/{num_inner_steps}, Loss: {loss.item():.4f}") # Log loss per step

    print(f"Adaptation to p={task_p} finished.")
    adapted_model.eval()

    # --- Meta-Test Evaluation ---
    if higher is not None:
        print("\n--- Evaluating Meta-Learned Models on Unseen Tasks (Meta-Test Set) ---")
        num_test_tasks_to_sample = 3
        if len(META_MODULI_TEST) >= num_test_tasks_to_sample:
            test_moduli_sample = random.sample(META_MODULI_TEST, num_test_tasks_to_sample)
        else:
            test_moduli_sample = META_MODULI_TEST
            print(f"Warning: Fewer than {num_test_tasks_to_sample} meta-test moduli available. Using all {len(test_moduli_sample)}.")

        meta_test_results = {
            "MLP": {p: {'pre_adapt_acc': -1.0, 'post_adapt_acc': -1.0} for p in test_moduli_sample},
            "CNN": {p: {'pre_adapt_acc': -1.0, 'post_adapt_acc': -1.0} for p in test_moduli_sample},
            "Transformer": {p: {'pre_adapt_acc': -1.0, 'post_adapt_acc': -1.0} for p in test_moduli_sample}
        }

        models_to_test = {
            "MLP": meta_model,
            "CNN": meta_model,
            "Transformer": meta_model
        }

        def evaluate_model(model, dataloader, task_p, device):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    relevant_outputs = outputs[:, :task_p]
                    _, predicted = torch.max(relevant_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return 100 * correct / total if total > 0 else 0.0

        for model_name, meta_model in models_to_test.items():
            print(f"\n-- Evaluating {model_name} on Meta-Test Tasks --")
            for p_test in test_moduli_sample:
                print(f"  Task: p = {p_test}")
                test_task_dataset = ModularAdditionDataset(p_test, meta_model.max_p_out)
                # Use full dataset for test evaluation, but could also use sample_task_batch query set
                test_task_loader = DataLoader(test_task_dataset, batch_size=STANDARD_BATCH_SIZE, shuffle=False)

                # Evaluate pre-adaptation
                pre_adapt_acc = evaluate_model(meta_model, test_task_loader, p_test, DEVICE)
                meta_test_results[model_name][p_test]['pre_adapt_acc'] = pre_adapt_acc
                print(f"    Accuracy (Pre-Adaptation): {pre_adapt_acc:.2f}%")

                # Adapt model
                adapted_model_test = adapt_maml_model(meta_model, p_test, K_SHOT, NUM_INNER_STEPS, INNER_LR, DEVICE)

                # Evaluate post-adaptation
                post_adapt_acc = evaluate_model(adapted_model_test, test_task_loader, p_test, DEVICE)
                meta_test_results[model_name][p_test]['post_adapt_acc'] = post_adapt_acc
                print(f"    Accuracy (Post-Adaptation): {post_adapt_acc:.2f}%")

        print("\n--- Meta-Test Evaluation Summary --- ")
        for model_name, results_by_p in meta_test_results.items():
            print(f"  {model_name}:")
            avg_pre = np.mean([res['pre_adapt_acc'] for res in results_by_p.values() if res['pre_adapt_acc'] >= 0])
            avg_post = np.mean([res['post_adapt_acc'] for res in results_by_p.values() if res['post_adapt_acc'] >= 0])
            print(f"    Average Pre-Adapt Acc: {avg_pre:.2f}%")
            print(f"    Average Post-Adapt Acc: {avg_post:.2f}%")
            for p_test, accs in results_by_p.items():
                 print(f"      p={p_test}: Pre={accs['pre_adapt_acc']:.2f}%, Post={accs['post_adapt_acc']:.2f}% (+{accs['post_adapt_acc']-accs['pre_adapt_acc']:.2f}%) ")

    else:
        print("\nSkipping Meta-Test Evaluation: 'higher' library not found or MAML training skipped.")

    return adapted_model


# --- Activation/Attention Extraction (Adapted) ---
def get_all_activations(model, dataloader, device, layer_names, task_p=None):
    """Extracts activations."""
    model.eval()
    model.to(device)
    if not hasattr(model, 'activations'): model.activations = {}
    if hasattr(model, '_register_hooks'): model._register_hooks()

    all_activations = {name: [] for name in layer_names}
    all_labels = []
    all_inputs = []

    with torch.no_grad():
        analysis_iterator = tqdm(dataloader, desc="Extracting Activations", leave=False)
        for inputs, labels in analysis_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)

            for name in layer_names:
                 # Adjust key based on how hooks store activations (e.g., name + '_output')
                 output_key = name # Default key
                 # Check for more specific keys if needed based on hook implementation
                 if name + '_output' in model.activations:
                      output_key = name + '_output'
                 elif name not in model.activations:
                      print(f"Warning: Act key '{name}' or '{name}_output' not found.")
                      continue # Skip if key not found

                 activation_data = model.activations[output_key].cpu().numpy()
                 if activation_data.ndim > 2:
                     if name.startswith("encoder_layer") and activation_data.shape[1] == 3:
                         # For transformer encoder layers, use the CLS token output
                         activation_data = activation_data[:, 0, :]
                     else:
                          # Flatten other multi-dim activations (e.g., CNN)
                          activation_data = activation_data.reshape(activation_data.shape[0], -1)
                 all_activations[name].append(activation_data) # Store with original layer name key

            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate
    concatenated_activations = {}
    for name in layer_names:
        if all_activations[name]:
             concatenated_activations[name] = np.concatenate(all_activations[name], axis=0)
        else:
             concatenated_activations[name] = np.array([])
             print(f"Warning: No activations collected for layer '{name}'.")

    if hasattr(model, 'remove_hooks'): model.remove_hooks()
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    all_inputs = np.concatenate(all_inputs, axis=0) if all_inputs else np.array([])
    return concatenated_activations, all_labels, all_inputs


def get_all_attention_weights(model, dataloader, device, layer_names):
    """ Extracts attention weights for Transformer. """
    if not isinstance(model, TransformerModel): return None
    model.eval(); model.to(device)
    all_attention = {name: [] for name in layer_names}
    with torch.no_grad():
        attn_iterator = tqdm(dataloader, desc="Extracting Attention Weights", leave=False)
        for inputs, _ in attn_iterator:
            _ = model(inputs.to(device))
            for name in layer_names:
                if name in model.attention_weights:
                    all_attention[name].append(model.attention_weights[name].cpu().numpy())
    concatenated_attention = {}
    for name in layer_names:
        if all_attention[name]: concatenated_attention[name] = np.concatenate(all_attention[name], axis=0)
        else: concatenated_attention[name] = None
    return concatenated_attention


# --- 4. Interpretability Functions (Mostly Unchanged) ---
# plot_dimensionality_reduction, compute_rdm, plot_rdm, compare_rdms,
# train_probe, plot_probe_results, plot_attention_map remain the same
# Minor robustness checks added previously are kept.

# --- 4.1 Dimensionality Reduction Visualization ---
def plot_dimensionality_reduction(activations, labels, method='tsne', n_components=2, title='Activation Visualization', color_label='Value', perplexity=30, n_iter=1000):
    if activations is None or activations.size == 0: print(f"Skipping plot '{title}': No data."); return
    if activations.ndim > 2: activations = activations.reshape(activations.shape[0], -1)
    num_samples = activations.shape[0]
    if num_samples == 0: print(f"Skipping plot '{title}': Zero samples."); return
    if labels.shape[0] != num_samples: print(f"Warning: Label/activation mismatch for '{title}'. Skipping."); return

    if method == 'tsne' and num_samples > 5000:
        print(f"Subsampling to 5000 points for TSNE plot: {title}")
        indices = np.random.choice(num_samples, 5000, replace=False)
        activations, labels = activations[indices], labels[indices]
        num_samples = 5000

    scaler = StandardScaler();
    try:
        scaled_activations = scaler.fit_transform(activations)
        if np.any(np.isnan(scaled_activations)) or np.any(np.isinf(scaled_activations)): # Handle potential NaNs/Infs after scaling
            print(f"Warning: NaNs/Infs found in scaled activations for '{title}'. Using nan_to_num.")
            scaled_activations = np.nan_to_num(scaled_activations)
        if np.all(np.var(scaled_activations, axis=0) < 1e-9): # Check for zero variance
             print(f"Warning: Near-zero variance in scaled activations for '{title}'. TSNE/PCA might be unstable.")

    except ValueError as e:
        print(f"Error scaling activations for '{title}': {e}. Skipping plot.")
        return


    if method == 'tsne':
        effective_perplexity = min(perplexity, max(1, num_samples - 2))
        reducer = TSNE(n_components=n_components, random_state=SEED, perplexity=effective_perplexity, n_iter=n_iter, init='pca', learning_rate='auto')
    elif method == 'pca': reducer = PCA(n_components=n_components, random_state=SEED)
    else: raise ValueError("Method must be 'tsne' or 'pca'")

    try: reduced_activations = reducer.fit_transform(scaled_activations)
    except Exception as e: print(f"Error during {method.upper()} for '{title}': {e}"); return

    plt.figure(figsize=(10, 8))
    if n_components == 2:
        scatter = plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=labels, cmap='hsv', alpha=0.7, s=10)
        plt.xlabel(f"{method.upper()} Comp 1"); plt.ylabel(f"{method.upper()} Comp 2")
    elif n_components == 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D # Import here
            ax = plt.axes(projection='3d')
            scatter = ax.scatter3D(reduced_activations[:, 0], reduced_activations[:, 1], reduced_activations[:, 2], c=labels, cmap='hsv', alpha=0.7, s=10)
            ax.set_xlabel(f"{method.upper()} Comp 1"); ax.set_ylabel(f"{method.upper()} Comp 2"); ax.set_zlabel(f"{method.upper()} Comp 3")
        except Exception as e:
             print(f"Could not create 3D plot for '{title}': {e}. Plotting 2D."); plt.close(); return # Close figure if 3D fails
    else: raise ValueError("n_components must be 2 or 3")

    plt.title(f"{title} ({method.upper()})")
    try: cbar = plt.colorbar(scatter); cbar.set_label(color_label)
    except Exception as e: print(f"Could not add colorbar for '{title}': {e}")
    plt.show()

# --- 4.2 Representational Similarity Analysis (RSA) ---
def compute_rdm(activations, metric='correlation'):
    if activations is None or activations.size == 0: return None, None
    if activations.ndim > 2: activations = activations.reshape(activations.shape[0], -1)
    num_samples = activations.shape[0]; subset_indices = None
    if num_samples == 0: return None, None
    if num_samples > 1000:
        print(f"Subsampling to 1000 points for RDM computation.")
        subset_indices = np.random.choice(num_samples, 1000, replace=False)
        activations = activations[subset_indices]; num_samples = 1000
    if num_samples < 2: return None, subset_indices
    scaler = StandardScaler(); scaled_activations = scaler.fit_transform(activations)
    if not np.all(np.var(scaled_activations, axis=0) > 1e-6): pass # Allow near-zero variance after scaling
    try:
        if np.any(np.isnan(scaled_activations)) or np.any(np.isinf(scaled_activations)): 
             print("Warning: NaNs/Infs in scaled activations for RDM. Using nan_to_num.")
             scaled_activations = np.nan_to_num(scaled_activations)

        distances = pdist(scaled_activations, metric=metric); rdm = squareform(distances)
        if np.any(np.isnan(rdm)) or np.any(np.isinf(rdm)): print(f"Warning: NaNs/Infs in computed RDM.")
        return rdm, subset_indices
    except Exception as e: print(f"Error computing RDM: {e}"); return None, subset_indices

def plot_rdm(rdm, title='RDM', labels=None, label_name='Value', cmap='viridis'):
    if rdm is None: print(f"Skipping RDM plot '{title}': No data."); return
    if np.any(np.isnan(rdm)) or np.any(np.isinf(rdm)): print(f"Warning: RDM '{title}' has NaNs/Infs.")
    plt.figure(figsize=(8, 7)); plot_labels = labels is not None and len(labels) == rdm.shape[0]
    if plot_labels:
        sort_indices = np.argsort(labels); rdm_sorted = rdm[sort_indices][:, sort_indices]
        cax = plt.imshow(rdm_sorted, cmap=cmap, interpolation='nearest')
        plt.title(f"{title} (Ordered by {label_name})")
        try:
            min_label, max_label = np.min(labels), np.max(labels); rdm_min, rdm_max = np.nanmin(rdm), np.nanmax(rdm)
            ticks = np.linspace(rdm_min, rdm_max, num=5) if not np.isnan(rdm_min) and not np.isnan(rdm_max) else None
            cbar = plt.colorbar(cax, ticks=ticks); cbar.set_label(f'Dissimilarity (Labels {min_label}-{max_label})')
        except Exception as e: print(f"Could not add sorted colorbar for '{title}': {e}"); plt.colorbar(cax, label='Dissimilarity')
    else:
        if labels is not None: print(f"Warning: Label/RDM mismatch for '{title}'. Plotting unordered.")
        cax = plt.imshow(rdm, cmap=cmap, interpolation='nearest'); plt.title(title); plt.colorbar(cax, label='Dissimilarity')
    plt.xlabel("Input Samples"); plt.ylabel("Input Samples"); plt.show()

def compare_rdms(rdm1, rdm2):
    if rdm1 is None or rdm2 is None: return None, None
    if rdm1.shape != rdm2.shape: print(f"RDM shape mismatch: {rdm1.shape} vs {rdm2.shape}"); return None, None
    if rdm1.shape[0] < 2: return None, None
    try:
        upper_tri_indices = np.triu_indices_from(rdm1, k=1); vec1 = rdm1[upper_tri_indices]; vec2 = rdm2[upper_tri_indices]
        finite_mask = np.isfinite(vec1) & np.isfinite(vec2)
        if not np.all(finite_mask): 
             print("Warning: Removing non-finite values for RDM comparison.")
             vec1, vec2 = vec1[finite_mask], vec2[finite_mask]
        if len(vec1) < 2: print("Warning: Less than 2 finite values for RDM comparison."); return 0.0, 1.0
        if np.var(vec1) < 1e-10 or np.var(vec2) < 1e-10: print("Warning: Near-zero variance in RDM vectors for comparison."); return 0.0, 1.0
        corr, p_value = spearmanr(vec1, vec2); return corr, p_value
    except Exception as e: print(f"Error comparing RDMs: {e}"); return None, None

# --- 4.3 Probing ---
def train_probe(activations, labels, test_size=0.3, random_state=SEED):
    if activations is None or activations.size == 0: return 0.0
    if labels is None or labels.size == 0: return 0.0
    if activations.shape[0] != labels.shape[0]: return 0.0
    if activations.ndim > 2: activations = activations.reshape(activations.shape[0], -1)
    if len(np.unique(labels)) < 2: print("Warning: Less than 2 classes for probing."); return 0.0
    try: X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=test_size, random_state=random_state, stratify=labels)
    except ValueError:
        print("Warning: Stratify failed in train_test_split, falling back to non-stratified.")
        try: X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=test_size, random_state=random_state)
        except Exception as e: print(f"Error in train_test_split: {e}"); return 0.0
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)): X_train_scaled = np.nan_to_num(X_train_scaled)
        if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)): X_test_scaled = np.nan_to_num(X_test_scaled)
    except ValueError as e: print(f"Error scaling probe data: {e}"); return 0.0
    probe = LogisticRegression(random_state=random_state, max_iter=1000, C=0.1, solver='liblinear', multi_class='ovr')
    try: probe.fit(X_train_scaled, y_train); accuracy = probe.score(X_test_scaled, y_test); return accuracy * 100
    except Exception as e: print(f"Error training/scoring probe: {e}"); return 0.0

def plot_probe_results(results_dict, title="Probing Results"):
    if not results_dict: print("No probe results to plot."); return
    filtered_results = {k: v for k, v in results_dict.items() if v is not None}
    if not filtered_results: print("No valid probe results to plot after filtering."); return
    labels = list(filtered_results.keys()); accuracies = list(filtered_results.values())
    plt.figure(figsize=(14, max(7, len(labels) * 0.4))); colors = sns.color_palette("viridis", len(labels))
    bars = plt.barh(labels, accuracies, color=colors); plt.xlabel("Probe Accuracy (%)"); plt.title(title); plt.xlim(0, 105); plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        if width > 0.1: plt.text(width + 1, bar.get_y() + bar.get_height()/2., f'{width:.1f}%', va='center', fontsize=8)
    plt.tight_layout(); plt.show()

# --- 4.4 Attention Map Visualization ---
def plot_attention_map(attention_weights, sample_idx, layer_idx, head_idx='mean', tokens=['CLS', 'a', 'b'], title=None):
    if attention_weights is None or attention_weights.ndim < 3: print(f"Skipping attention plot {title}: Invalid weights shape."); return
    if sample_idx >= attention_weights.shape[0]: print(f"Skipping attention plot {title}: Sample index {sample_idx} out of bounds."); return
    
    # Assuming attention_weights shape is (batch, num_heads, query_len, key_len)
    # Or (batch, query_len, key_len) if average_attn_weights=True
    if attention_weights.ndim == 4: 
         if head_idx == 'mean':
             attn_sample = attention_weights[sample_idx].mean(axis=0) # Average over heads
         elif isinstance(head_idx, int) and 0 <= head_idx < attention_weights.shape[1]:
             attn_sample = attention_weights[sample_idx, head_idx]
         else:
             print(f"Skipping attention plot {title}: Invalid head_idx {head_idx}. Using mean.")
             attn_sample = attention_weights[sample_idx].mean(axis=0)
    elif attention_weights.ndim == 3:
        attn_sample = attention_weights[sample_idx]
    else:
         print(f"Skipping attention plot {title}: Unexpected weights dimension {attention_weights.ndim}.")
         return
         
    if np.any(np.isnan(attn_sample)) or np.any(np.isinf(attn_sample)): 
        print(f"Warning: NaNs/Infs in attention map for {title}. Using nan_to_num.")
        attn_sample = np.nan_to_num(attn_sample)
        
    expected_len = len(tokens)
    # Adjust check for potentially averaged weights or specific head
    if attn_sample.shape[0] != expected_len or attn_sample.shape[1] != expected_len: 
         print(f"Skipping attention plot {title}: Shape mismatch. Expected ({expected_len},{expected_len}), Got {attn_sample.shape}.")
         return
         
    plt.figure(figsize=(6, 5)); sns.heatmap(attn_sample, annot=True, fmt=".2f", cmap="Blues", square=True, linewidths=.5, linecolor='black', xticklabels=tokens, yticklabels=tokens, cbar=False)
    if title is None: title = f"Avg. Attention Map (Layer {layer_idx}, Sample {sample_idx})"
    plt.title(title, fontsize=10); plt.xlabel("Key Tokens (Attended To)"); plt.ylabel("Query Tokens (Attending From)"); plt.xticks(rotation=0, fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.show()


# --- 5. Main Execution Block ---
if __name__ == "__main__":

    # --- Setup ---
    print("--- MAML Experiment Script ---")
    # Standard Dataloaders (fixed p - needed for analysis)
    standard_dataset_for_analysis = ModularAdditionDataset(STANDARD_MODULUS, MAX_MODULUS)
    # We need a loader for the STANDARD_MODULUS task for analysis purposes
    # Create a loader using the full dataset, as split is irrelevant here
    std_test_loader = DataLoader(standard_dataset_for_analysis, batch_size=STANDARD_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if DEVICE=='cuda' else False)
    print(f"Created analysis DataLoader for standard task (p={STANDARD_MODULUS}, size={len(standard_dataset_for_analysis)}) using full dataset.")

    # --- Load Pre-Trained Standard Models ---
    print("\n--- Loading Pre-Trained Standard Models ---")
    mlp_model_std = MLP(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    cnn_model_std = CNN1D(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    transformer_model_std = TransformerModel(MAX_MODULUS, INPUT_PROJ_DIM, nhead=N_HEAD, num_encoder_layers=NUM_TF_ENC_LAYERS).to(DEVICE)

    models_to_load = {
        "MLP_Std": mlp_model_std,
        "CNN_Std": cnn_model_std,
        "Transformer_Std": transformer_model_std
    }

    for model_key, model_instance in models_to_load.items():
        model_path = os.path.join(MODEL_SAVE_DIR, f"{model_key}_standard_p{STANDARD_MODULUS}_best.pth")
        if os.path.exists(model_path):
            try:
                model_instance.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model_instance.eval() # Set to evaluation mode
                print(f"Successfully loaded standard model weights for {model_key} from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_key} from {model_path}: {e}")
                print(f"Ensure '{model_path}' exists and was generated by the vanilla training script.")
                # Decide how to handle error: exit, continue with untrained model? For now, continue.
        else:
            print(f"Warning: Standard model file not found at {model_path}. Analysis will use an untrained standard model for {model_key}.")
            print("Please run the modular_arithmetic_vanilla.py script first.")

    # --- Instantiate MAML Models ---
    print("\n--- Instantiating MAML Models ---")
    mlp_model_maml = MLP(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    cnn_model_maml = CNN1D(MAX_MODULUS, INPUT_PROJ_DIM, HIDDEN_DIM).to(DEVICE)
    transformer_model_maml = TransformerModel(MAX_MODULUS, INPUT_PROJ_DIM, nhead=N_HEAD, num_encoder_layers=NUM_TF_ENC_LAYERS).to(DEVICE)

    # Define criterion (needed for MAML adaptation/evaluation)
    criterion = nn.CrossEntropyLoss()

    # Meta-Optimizers
    meta_optimizer_mlp = optim.Adam(mlp_model_maml.parameters(), lr=META_LR)
    meta_optimizer_cnn = optim.Adam(cnn_model_maml.parameters(), lr=META_LR)
    meta_optimizer_transformer = optim.Adam(transformer_model_maml.parameters(), lr=META_LR)

    # --- Train or Load MAML Meta-Models ---
    # Individual loading flags
    mlp_maml_loaded = False
    cnn_maml_loaded = False
    tf_maml_loaded = False

    if higher is not None:
        print("\n--- Checking for and Attempting to Load Existing MAML Meta-Models ---")
        mlp_maml_path = os.path.join(MODEL_SAVE_DIR, f"MLP_MAML_maml_meta_best.pth")
        cnn_maml_path = os.path.join(MODEL_SAVE_DIR, f"CNN_MAML_maml_meta_best.pth")
        tf_maml_path = os.path.join(MODEL_SAVE_DIR, f"Transformer_MAML_maml_meta_best.pth")

        # --- Attempt Loading MLP MAML ---
        print(f"Checking for MLP MAML model at: {os.path.abspath(mlp_maml_path)}")
        if os.path.exists(mlp_maml_path):
            try:
                print(f"  Attempting to load MLP MAML from {mlp_maml_path}...")
                mlp_model_maml.load_state_dict(torch.load(mlp_maml_path, map_location=DEVICE))
                mlp_model_maml.eval()
                print("  Successfully loaded MLP MAML.")
                mlp_maml_loaded = True
            except Exception as e:
                print(f"  ERROR loading MLP MAML: {e}. Will retrain.")
        else:
            print("  MLP MAML model file not found. Will train.")

        # --- Attempt Loading CNN MAML ---
        print(f"Checking for CNN MAML model at: {os.path.abspath(cnn_maml_path)}")
        if os.path.exists(cnn_maml_path):
            try:
                print(f"  Attempting to load CNN MAML from {cnn_maml_path}...")
                cnn_model_maml.load_state_dict(torch.load(cnn_maml_path, map_location=DEVICE))
                cnn_model_maml.eval()
                print("  Successfully loaded CNN MAML.")
                cnn_maml_loaded = True
            except Exception as e:
                print(f"  ERROR loading CNN MAML: {e}. Will retrain.")
        else:
            print("  CNN MAML model file not found. Will train.")

        # --- Attempt Loading Transformer MAML ---
        print(f"Checking for Transformer MAML model at: {os.path.abspath(tf_maml_path)}")
        if os.path.exists(tf_maml_path):
            try:
                print(f"  Attempting to load Transformer MAML from {tf_maml_path}...")
                transformer_model_maml.load_state_dict(torch.load(tf_maml_path, map_location=DEVICE))
                transformer_model_maml.eval()
                print("  Successfully loaded Transformer MAML.")
                tf_maml_loaded = True
            except Exception as e:
                print(f"  ERROR loading Transformer MAML: {e}. Will retrain.")
        else:
            print("  Transformer MAML model file not found. Will train.")

        # --- Conditionally Train MAML Models ---
        if not mlp_maml_loaded or not cnn_maml_loaded or not tf_maml_loaded:
            print("\n--- Starting MAML Meta-Model Training (for models not loaded) ---")
            if not mlp_maml_loaded:
                print("Training MLP MAML...")
                mlp_model_maml = meta_train_model(mlp_model_maml, META_MODULI_TRAIN, K_SHOT, NUM_INNER_STEPS, INNER_LR, meta_optimizer_mlp, META_NUM_EPOCHS, DEVICE, "MLP_MAML", MODEL_SAVE_DIR, META_BATCH_SIZE)
                mlp_maml_loaded = True # Mark as loaded/trained now
            if not cnn_maml_loaded:
                print("Training CNN MAML...")
                cnn_model_maml = meta_train_model(cnn_model_maml, META_MODULI_TRAIN, K_SHOT, NUM_INNER_STEPS, INNER_LR, meta_optimizer_cnn, META_NUM_EPOCHS, DEVICE, "CNN_MAML", MODEL_SAVE_DIR, META_BATCH_SIZE)
                cnn_maml_loaded = True # Mark as loaded/trained now
            if not tf_maml_loaded:
                print("Training Transformer MAML...")
                transformer_model_maml = meta_train_model(transformer_model_maml, META_MODULI_TRAIN, K_SHOT, NUM_INNER_STEPS, INNER_LR, meta_optimizer_transformer, META_NUM_EPOCHS, DEVICE, "Transformer_MAML", MODEL_SAVE_DIR, META_BATCH_SIZE)
                tf_maml_loaded = True # Mark as loaded/trained now
            print("\n--- MAML Meta-Model Training Finished (for required models) ---")
        else:
            print("\n--- All MAML models successfully loaded, skipping training. ---")

        # --- Adapt and Analyze --- (Only if MAML models were trained or loaded successfully)
        # Combine flags to check if analysis can proceed for all
        maml_models_trained_or_loaded = mlp_maml_loaded and cnn_maml_loaded and tf_maml_loaded
        if maml_models_trained_or_loaded:
            print("\n--- Running Interpretability Analysis (Adaptation + Activation Extraction) ---")
            mlp_model_maml_adapted = adapt_maml_model(mlp_model_maml, STANDARD_MODULUS, K_SHOT, NUM_INNER_STEPS, INNER_LR, DEVICE)
            cnn_model_maml_adapted = adapt_maml_model(cnn_model_maml, STANDARD_MODULUS, K_SHOT, NUM_INNER_STEPS, INNER_LR, DEVICE)
            transformer_model_maml_adapted = adapt_maml_model(transformer_model_maml, STANDARD_MODULUS, K_SHOT, NUM_INNER_STEPS, INNER_LR, DEVICE)
        else:
            # This case should ideally not be reached if loading fails and higher is None, but handle defensively
            print("\nSkipping Adaptation and Analysis: MAML models could not be loaded or trained.")
            # Create dummy adapted models if needed downstream, or handle appropriately
            mlp_model_maml_adapted = copy.deepcopy(mlp_model_maml)
            cnn_model_maml_adapted = copy.deepcopy(cnn_model_maml)
            transformer_model_maml_adapted = copy.deepcopy(transformer_model_maml)

    else: # Case where higher is not installed
        print("\n--- Skipping MAML Training and Analysis: 'higher' library not found ---")
        # Need placeholder adapted models if analysis code expects them
        mlp_model_maml_adapted = copy.deepcopy(mlp_model_maml)
        cnn_model_maml_adapted = copy.deepcopy(cnn_model_maml)
        transformer_model_maml_adapted = copy.deepcopy(transformer_model_maml)
        maml_models_trained_or_loaded = False # Explicitly set flag


    # Define layers to analyze
    # Adjust names based on hook registration (e.g., adding '_input' or '_output')
    mlp_layers = ['input_proj_output', 'linear_1_output', 'linear_2_output', 'pre_softmax_output']
    cnn_layers = ['input_proj_output', 'conv_1_output', 'dense_1_output', 'pre_softmax_output']
    # For Transformer, remove input_proj_output, hook names are like 'encoder_layer_0_output'
    tf_layers = [f'encoder_layer_{i}_output' for i in range(NUM_TF_ENC_LAYERS)] + ['pre_softmax_output']
    tf_attn_keys = [f'encoder_layer_{i}_attention' for i in range(NUM_TF_ENC_LAYERS)]


    # --- Extract Activations and Attention ---
    # Use the std_test_loader (full dataset for p=97) for all analysis
    print("\nExtracting activations for ALL model states...")
    # Standard models (Loaded)
    mlp_act_std, lbl_std, inp_std = get_all_activations(mlp_model_std, std_test_loader, DEVICE, mlp_layers, task_p=STANDARD_MODULUS)
    cnn_act_std, _, _ = get_all_activations(cnn_model_std, std_test_loader, DEVICE, cnn_layers, task_p=STANDARD_MODULUS)
    tf_act_std, _, _ = get_all_activations(transformer_model_std, std_test_loader, DEVICE, tf_layers, task_p=STANDARD_MODULUS)
    tf_attn_std = get_all_attention_weights(transformer_model_std, std_test_loader, DEVICE, tf_attn_keys)

    # MAML meta models (Before adaptation on p=97)
    mlp_act_meta, _, _ = get_all_activations(mlp_model_maml, std_test_loader, DEVICE, mlp_layers, task_p=STANDARD_MODULUS)
    cnn_act_meta, _, _ = get_all_activations(cnn_model_maml, std_test_loader, DEVICE, cnn_layers, task_p=STANDARD_MODULUS)
    tf_act_meta, _, _ = get_all_activations(transformer_model_maml, std_test_loader, DEVICE, tf_layers, task_p=STANDARD_MODULUS)
    tf_attn_meta = get_all_attention_weights(transformer_model_maml, std_test_loader, DEVICE, tf_attn_keys)

    # MAML adapted models (After adaptation on p=97)
    mlp_act_adapt, _, _ = get_all_activations(mlp_model_maml_adapted, std_test_loader, DEVICE, mlp_layers, task_p=STANDARD_MODULUS)
    cnn_act_adapt, _, _ = get_all_activations(cnn_model_maml_adapted, std_test_loader, DEVICE, cnn_layers, task_p=STANDARD_MODULUS)
    tf_act_adapt, _, _ = get_all_activations(transformer_model_maml_adapted, std_test_loader, DEVICE, tf_layers, task_p=STANDARD_MODULUS)
    tf_attn_adapt = get_all_attention_weights(transformer_model_maml_adapted, std_test_loader, DEVICE, tf_attn_keys)

    test_labels = lbl_std; test_inputs = inp_std # Use labels/inputs from standard task analysis


    # --- Analysis Sections (Dimensionality Reduction, RSA, Probing, Attention) ---
    print("\n--- Plotting Dimensionality Reduction (Comparison) ---")
    layer_key_to_plot = 'pre_softmax_output' # Key used in get_all_activations dictionary

    # Check if activations were actually generated (depends on maml_models_trained_or_loaded)
    plot_tsne = maml_models_trained_or_loaded or os.path.exists(os.path.join(MODEL_SAVE_DIR, f"MLP_Std_standard_p{STANDARD_MODULUS}_best.pth"))

    if plot_tsne:
        # MLP Plots
        if layer_key_to_plot in mlp_act_std: plot_dimensionality_reduction(mlp_act_std.get(layer_key_to_plot), test_labels, method='tsne', title=f'MLP Standard (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in mlp_act_meta: plot_dimensionality_reduction(mlp_act_meta.get(layer_key_to_plot), test_labels, method='tsne', title=f'MLP MAML-Meta Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in mlp_act_adapt: plot_dimensionality_reduction(mlp_act_adapt.get(layer_key_to_plot), test_labels, method='tsne', title=f'MLP MAML-Adapted (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')

        # CNN Plots
        if layer_key_to_plot in cnn_act_std: plot_dimensionality_reduction(cnn_act_std.get(layer_key_to_plot), test_labels, method='tsne', title=f'CNN Standard (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in cnn_act_meta: plot_dimensionality_reduction(cnn_act_meta.get(layer_key_to_plot), test_labels, method='tsne', title=f'CNN MAML-Meta Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in cnn_act_adapt: plot_dimensionality_reduction(cnn_act_adapt.get(layer_key_to_plot), test_labels, method='tsne', title=f'CNN MAML-Adapted (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')

        # Transformer Plots
        if layer_key_to_plot in tf_act_std: plot_dimensionality_reduction(tf_act_std.get(layer_key_to_plot), test_labels, method='tsne', title=f'Transformer Standard (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in tf_act_meta: plot_dimensionality_reduction(tf_act_meta.get(layer_key_to_plot), test_labels, method='tsne', title=f'Transformer MAML-Meta Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')
        if maml_models_trained_or_loaded and layer_key_to_plot in tf_act_adapt: plot_dimensionality_reduction(tf_act_adapt.get(layer_key_to_plot), test_labels, method='tsne', title=f'Transformer MAML-Adapted (p={STANDARD_MODULUS}) Pre-Softmax', color_label=f'Result mod {STANDARD_MODULUS}')

        # ... (Rest of RSA, Probing, Attention analysis as needed) ...
        print("\n--- T-SNE Analysis Finished ---")
    else:
        print("\nSkipping T-SNE plots: MAML models not loaded/trained and/or standard models not found.")

    print("\n--- Experiment Finished ---") 