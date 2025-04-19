# -*- coding: utf-8 -*-
"""
Experiment comparing internal representations of MLP, CNN, and Transformer
on a few-shot concept learning task using MAML.

Adapted from the ManyPaths project and modular_arithmetic_maml.py.
Includes T-SNE visualization of meta-learned and adapted representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm.notebook import tqdm # Use tqdm if running in notebook
except ImportError:
    print("tqdm not found. Install with 'pip install tqdm' for progress bars.")
    def tqdm(iterable, **kwargs): # Dummy tqdm if not installed
        return iterable
# Remove learn2learn import for vanilla training
# try:
#     import learn2learn as l2l # learn2learn for MAML algorithms
# except ImportError:
#     print("learn2learn not found. Install with 'pip install learn2learn'. MAML training will fail.")
#     l2l = None

# Imports potentially needed by copied classes/functions from ManyPaths
import os
import random
import math
from typing import Optional, Tuple, List, Dict
from torch import Tensor
import copy
from PIL import Image
from torchvision import transforms # May be needed by dataset if using images directly
from collections import defaultdict
# Assuming generate_concepts.py and grammer.py are findable via sys path or relative import
# If not, these might need to be copied or paths adjusted
# from generate_concepts import generate_concept # Used by MetaBitConceptsDataset (image)
# from grammer import DNFHypothesis # Used by MetaBitConceptsDataset

# --- Configuration ---
# General
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE) # Optional MPS check
print(f"Using device: {DEVICE}")
MODEL_SAVE_DIR = "saved_models_concept_maml"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
DATA_TYPE = "image" # ["image", "bits"] - determines input format for models
N_FEATURES = 4 # Number of binary features defining concepts

# Model Architecture Defaults (Inspired by 'medium' size in ManyPaths/visualize_concepts_simplified.py)
# These might be adjusted or loaded from a config if needed
MLP_CONFIG = {"n_hidden": 64, "n_layers": 4}
CNN_CONFIG = {"n_hiddens": [32, 32, 16], "n_layers": 4} # n_hiddens list for CNN
TRANSFORMER_CONFIG = {"d_model": 64, "nhead": 4, "num_layers": 2}

# Concept Task Specifics
N_OUTPUT = 1 # Binary classification output
CHANNELS = 3 if DATA_TYPE == "image" else 1 # Input channels
BITS = N_FEATURES if DATA_TYPE == "bits" else None # Input size for bits MLP

# Add Standard Training Config
STANDARD_EPOCHS = 100 # Adjust as needed
STANDARD_BATCH_SIZE = 16 # Concepts dataset is small
STANDARD_LEARNING_RATE = 1e-3
# LR_SCHEDULER_STEP = 10 # Optional: Add scheduler later if needed
# LR_SCHEDULER_GAMMA = 0.5
VALIDATION_SPLIT = 0.5 # Proportion of the single task's data for validation (Increased to 50%)

# For reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Constants from ManyPaths/constants.py needed for Dataset
FEATURE_VALUES = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=int,
)

# Placeholder for DNFHypothesis if grammer.py is complex/not needed immediately
class DNFHypothesis:
    # Simplified placeholder - assumes a random function generation
    def __init__(self, n_features=4, no_true_false_top=True, b=1.0):
        self.n_features = n_features
        # Define a random but fixed rule based on seed for reproducibility within run
        self.rule_indices = random.sample(range(n_features), k=random.randint(1, n_features))
        self.rule_values = [random.choice([0, 1]) for _ in self.rule_indices]
        print(f"  [Placeholder DNF] Generated random rule: Features {self.rule_indices} must be {self.rule_values}")

    def function(self, features):
        # Simple conjunction based on the random rule
        match = True
        for idx, val in zip(self.rule_indices, self.rule_values):
            if features[idx] != val:
                match = False
                break
        return 1.0 if match else 0.0

# generate_concept function (from ManyPaths/visualize_concepts_simplified.py)
def generate_concept(bits, scale=1.0):
    """Generate a simple concept image based on 4 bits."""
    if not (len(bits) == 4):
        raise ValueError("Bits must be length 4.")

    # Initialize a blank grid
    grid_image = np.ones((32, 32, 3), dtype=np.float32) * 255

    # Extract bits
    color = (1, 2) if bits[0] == 1 else (0, 1)
    shape = bits[1] == 1
    size = 4 if bits[2] == 1 else 10
    style = bits[3] == 1

    if shape:
        grid_image[size : 32 - size, size : 32 - size, color] = 0
        if style == 1:
            grid_image[size : 32 - size, size : 32 - size : 2, color] = 200
    else:
        for i in range(32 - 2 * size):
            grid_image[
                32 - (size + i + 1), i // 2 + size : 32 - i // 2 - size, color
            ] = 0
        if style == 1:
            for i in range(0, 32, 1):
                for j in range(0, 32, 2):
                    if grid_image[i, j, color].any() == 0:
                        grid_image[i, j, color] = 200

    grid_image = grid_image / scale
    return grid_image

# --- 1. Data Generation (MetaBitConceptsDataset) ---
# Copied and adapted from ManyPaths/datasets.py
class BaseMetaDataset(Dataset):
    def __init__(self):
        self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # Returns: Support_X, Support_y, Query_X, Query_y
        # Original task tuple: (X_image_s, X_s, y_s, X_image_q, X_q, y_q, n_support)
        # Or for bits: (X_bits_s, X_s, y_s, X_bits_q, X_q, y_q, n_support)
        task_data = self.tasks[idx]
        X_s = task_data[0] # Input data (image patches, bits, or numbers)
        y_s = task_data[2]
        X_q = task_data[3]
        y_q = task_data[5]
        return X_s, y_s, X_q, y_q

    def _image_to_patches(self, image_batch, patch_size=4):
        # Converts images to sequences of patches for sequence models (LSTM/Transformer)
        B, C, H, W = image_batch.shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Image dimensions ({H}x{W}) must be divisible by patch size ({patch_size})")
        patches = image_batch.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        # Reshape and permute: (B, C, NumPatchesH, NumPatchesW, PatchH, PatchW) -> (B, NumPatches, C*PatchH*PatchW)
        patches = patches.reshape(B, C, -1, patch_size, patch_size).permute(
            0, 2, 1, 3, 4
        )
        # (B, NumPatches, FeaturesPerPatch)
        return patches.reshape(B, -1, C * patch_size * patch_size)


class MetaBitConceptsDataset(BaseMetaDataset):
    def __init__(
        self,
        n_tasks: int = 10000,
        data: str = "image", # Should match global DATA_TYPE
        model: str = "cnn", # Model type affects data format (patches vs flat)
        k_shot_min: int = 2, # Min K for support set size
        k_shot_max: int = 20, # Max K for support set size (original used randint(2,20))
        fixed_k_shot: int = None, # Set to override random k per task
        patch_size: int = 4, # Used if data="image" and model is LSTM/Transformer
    ):
        super().__init__()
        print(f"Initializing MetaBitConceptsDataset: n_tasks={n_tasks}, data={data}, model={model}")
        if data not in ["image", "bits"]:
            raise ValueError("Data type must be 'image' or 'bits' for MetaBitConceptsDataset")
        self.n_tasks = n_tasks
        self.data = data
        self.model = model
        self.k_shot_min = k_shot_min
        self.k_shot_max = k_shot_max
        self.fixed_k_shot = fixed_k_shot
        self.patch_size = patch_size
        self._generate_tasks()

    def _generate_tasks(self):
        print(f"Generating {self.n_tasks} concept learning tasks...")
        if self.data == "image":
            self._generate_image_tasks()
        else:
            self._generate_bit_tasks()
        print(f"Generated {len(self.tasks)} tasks successfully.")

    def _generate_image_tasks(self):
        # Pre-generate all 16 concept images
        X_q_base = torch.tensor(FEATURE_VALUES, dtype=torch.float)
        all_images_np = [generate_concept(bits, scale=1.0).transpose(2, 0, 1) for bits in FEATURE_VALUES] # Transpose C,H,W
        X_image_q_all = torch.tensor(np.array(all_images_np), dtype=torch.float)

        # Normalize images (using statistics across all 16 images)
        mean = X_image_q_all.mean(dim=[0, 2, 3]) # (C,)
        std = X_image_q_all.std(dim=[0, 2, 3]) # (C,)
        std[std == 0] = 1.0 # Avoid division by zero if a channel is constant
        transform = transforms.Normalize(mean=mean, std=std)
        X_image_q_all_norm = transform(X_image_q_all) # Shape: (16, C, H, W)

        # Pre-process query set based on model type
        if self.model == "mlp":
            # Flatten images: (16, C*H*W)
            X_q_processed = X_image_q_all_norm.reshape(16, -1)
        elif self.model in ["lstm", "transformer"]:
            # Convert images to patches: (16, NumPatches, FeaturesPerPatch)
             try:
                 X_q_processed = self._image_to_patches(X_image_q_all_norm, patch_size=self.patch_size)
             except ValueError as e:
                 print(f"Error creating patches for query set: {e}")
                 return # Cannot proceed if patching fails
        else: # Assume CNN
            # Keep as images: (16, C, H, W)
            X_q_processed = X_image_q_all_norm

        # Generate tasks by sampling hypotheses and support sets
        while len(self.tasks) < self.n_tasks:
            # Generate a random concept rule (hypothesis)
            hyp = DNFHypothesis(n_features=N_FEATURES) # Using placeholder
            labels = torch.tensor([hyp.function(f) for f in FEATURE_VALUES], dtype=torch.float).unsqueeze(1)

            # Ensure the hypothesis isn't trivial (all 0s or all 1s)
            if torch.all(labels == 0.0) or torch.all(labels == 1.0):
                # print("  Skipping trivial hypothesis (all same labels).")
                continue

            # Determine k for this task
            if self.fixed_k_shot is not None:
                k_shot = self.fixed_k_shot
            else:
                k_shot = random.randint(self.k_shot_min, self.k_shot_max)

            # Sample support indices (k examples)
            # Ensure at least one example from each class if possible
            pos_indices = torch.where(labels == 1.0)[0]
            neg_indices = torch.where(labels == 0.0)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                 print(f"Warning: Hypothesis resulted in only one class. Skipping.") # Should not happen with check above
                 continue

            support_indices = []
            # Try to sample floor(k/2) positive and ceil(k/2) negative, handle insufficient samples
            n_pos = k_shot // 2
            n_neg = k_shot - n_pos

            pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
            neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

            support_indices = torch.cat((pos_sample_indices, neg_sample_indices))

            # If we couldn't get k samples (e.g., k=10 but only 3 positive examples exist),
            # fill remaining slots from the other class if possible.
            remaining_k = k_shot - len(support_indices)
            if remaining_k > 0:
                # print(f"  Adjusting support set sampling due to class imbalance for k={k_shot}")
                if len(pos_sample_indices) < n_pos:
                     fill_indices = neg_indices[torch.randperm(len(neg_indices))[:remaining_k]]
                else: # len(neg_sample_indices) < n_neg
                     fill_indices = pos_indices[torch.randperm(len(pos_indices))[:remaining_k]]
                support_indices = torch.cat((support_indices, fill_indices))

            # Final check if we still don't have k_shot (highly unlikely unless k is large and one class is tiny)
            if len(support_indices) < k_shot:
                 print(f"Warning: Could not sample k={k_shot} examples for hypothesis. Got {len(support_indices)}. Skipping task.")
                 continue

            # Shuffle final support indices
            support_indices = support_indices[torch.randperm(len(support_indices))]

            # Select support data based on model type
            y_s = labels[support_indices]
            if self.model == "mlp":
                X_s_processed = X_q_processed[support_indices] # Flattened
            elif self.model in ["lstm", "transformer"]:
                X_s_processed = X_q_processed[support_indices] # Patches
            else: # CNN
                X_s_processed = X_image_q_all_norm[support_indices] # Images

            # Store the task (Support X, Support y, Query X, Query y)
            # Query y is the full set of labels for all 16 concepts
            self.tasks.append((X_s_processed, y_s, X_q_processed, labels))

    def _generate_bit_tasks(self):
        # Base features (0/1)
        X_q_base = torch.tensor(FEATURE_VALUES, dtype=torch.float)
        # Input features (-1/1)
        X_q_bits = X_q_base * 2.0 - 1.0 # Shape: (16, 4)

        # Pre-process query set based on model type
        if self.model in ["lstm", "transformer"]:
            # Add sequence dimension: (16, 4, 1) -> (Batch, SeqLen, Features)
            X_q_processed = X_q_bits.unsqueeze(-1)
        else: # MLP/CNN (CNN might not make sense for bits, but MLP does)
            # Keep as: (16, 4)
            X_q_processed = X_q_bits

        # Generate tasks by sampling hypotheses and support sets
        while len(self.tasks) < self.n_tasks:
            hyp = DNFHypothesis(n_features=N_FEATURES)
            labels = torch.tensor([hyp.function(f) for f in FEATURE_VALUES], dtype=torch.float).unsqueeze(1)

            if torch.all(labels == 0.0) or torch.all(labels == 1.0):
                continue # Skip trivial

            if self.fixed_k_shot is not None:
                k_shot = self.fixed_k_shot
            else:
                k_shot = random.randint(self.k_shot_min, self.k_shot_max)

            # Sample support indices (similar logic as image task)
            pos_indices = torch.where(labels == 1.0)[0]
            neg_indices = torch.where(labels == 0.0)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0: continue

            n_pos = k_shot // 2
            n_neg = k_shot - n_pos
            pos_sample_indices = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
            neg_sample_indices = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]
            support_indices = torch.cat((pos_sample_indices, neg_sample_indices))

            remaining_k = k_shot - len(support_indices)
            if remaining_k > 0:
                if len(pos_sample_indices) < n_pos:
                     fill_indices = neg_indices[torch.randperm(len(neg_indices))[:remaining_k]]
                else:
                     fill_indices = pos_indices[torch.randperm(len(pos_indices))[:remaining_k]]
                support_indices = torch.cat((support_indices, fill_indices))

            if len(support_indices) < k_shot: continue # Skip if still couldn't sample k

            support_indices = support_indices[torch.randperm(len(support_indices))]

            # Select support data based on model type
            y_s = labels[support_indices]
            X_s_processed = X_q_processed[support_indices]

            # Store the task (Support X, Support y, Query X, Query y)
            self.tasks.append((X_s_processed, y_s, X_q_processed, labels))

# --- 2. Model Definitions (MLP, CNN, Transformer from ManyPaths + Hooks) ---

# --- Positional Encoding (Copied from ManyPaths/models.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register buffer requires pe to be contiguous in memory
        self.register_buffer("pe", pe.unsqueeze(0).contiguous())

    def forward(self, x):
        seq_len = x.size(1)
        # Ensure pe is compatible with x's device
        pos_encoding = self.pe[:, :seq_len].requires_grad_(False).to(x.device)
        return x + pos_encoding

# --- Transformer Encoder Layer (Copied from ManyPaths/models.py) ---
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.0, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Note: ManyPaths uses norm_first=True logic (norm -> attn/ffwd -> residual)
        src2 = self.norm1(src)
        attn_output, attn_weights = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
            need_weights=True, average_attn_weights=False # Get per-head weights if needed
        )
        src = src + attn_output # Residual connection 1
        src2 = self.norm2(src) # Norm before FFN
        ffn_output = self.linear2(torch.relu(self.linear1(src2)))
        src = src + ffn_output # Residual connection 2
        # Return attention weights along with output for potential analysis
        return src, attn_weights

# --- MLP Model (Copied from ManyPaths/models.py and Hooked) ---
class ConceptMLP(nn.Module):
    def __init__(
        self,
        n_input: int, # Determined by DATA_TYPE (flattened image or bits)
        n_output: int = N_OUTPUT,
        n_hidden: int = MLP_CONFIG["n_hidden"],
        n_layers: int = MLP_CONFIG["n_layers"],
        n_input_channels: int = CHANNELS, # Needed if input size < 64
    ):
        super().__init__()
        print(f"Initializing ConceptMLP: n_input={n_input}, n_output={n_output}, n_hidden={n_hidden}, n_layers={n_layers}")
        layers = []
        # ManyPaths MLP has a specific structure based on input size
        if n_input < 64:
            # Input projection for small inputs (like bits)
            self.input_layer = nn.Linear(n_input, 32 * 32 * n_input_channels)
            layers.extend([
                nn.BatchNorm1d(32 * 32 * n_input_channels), nn.ReLU(),
                nn.Linear(32 * 32 * n_input_channels, n_hidden),
                nn.BatchNorm1d(n_hidden), nn.ReLU()
            ])
            current_dim = n_hidden
            self.hidden_layers_start_idx = 4 # Index of first hidden Linear layer in nn.Sequential
        else:
            # Direct input for larger inputs (like flattened images)
            self.input_layer = nn.Linear(n_input, n_hidden)
            layers.extend([
                nn.BatchNorm1d(n_hidden), nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden), nn.ReLU()
            ])
            current_dim = n_hidden
            self.hidden_layers_start_idx = 3 # Index of first hidden Linear layer

        # Add remaining hidden layers
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(current_dim, n_hidden), nn.BatchNorm1d(n_hidden), nn.ReLU(),
            ])
            current_dim = n_hidden

        # Final output layer
        self.output_layer = nn.Linear(current_dim, n_output)

        # Store layers for forward pass and hooking
        self.layers = nn.Sequential(*layers)

        # --- Hooking Attributes ---
        self.activations = {}
        self._hook_handles = []

    # --- Hooking Methods --- 
    def _get_activation(self, name):
        def hook(model, input, output):
            # Detach and clone to prevent memory issues and graph modifications
            act_input = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
            act_output = output.detach().clone()
            self.activations[name + '_input'] = act_input
            self.activations[name + '_output'] = act_output
        return hook

    def _register_hooks(self):
        self.remove_hooks()
        # Hook input layer output
        self._hook_handles.append(self.input_layer.register_forward_hook(self._get_activation('input_layer')))
        # Hook first hidden linear layer output (after ReLU)
        first_hidden_linear_idx = self.hidden_layers_start_idx
        if len(self.layers) > first_hidden_linear_idx + 2: # Ensure layer exists
             # Hook output of ReLU after the first hidden Linear
             self._hook_handles.append(self.layers[first_hidden_linear_idx + 2].register_forward_hook(self._get_activation('hidden_1')))
        # Hook pre-output layer (input to final nn.Linear)
        # The input to the final layer is the output of the last ReLU in self.layers
        if len(self.layers) > 0: 
             self._hook_handles.append(self.layers[-1].register_forward_hook(self._get_activation('pre_output')))
        # Hook final output
        self._hook_handles.append(self.output_layer.register_forward_hook(self._get_activation('output')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}
    # --- End Hooking Methods --- 

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x

# --- CNN Model (Copied from ManyPaths/models.py and Hooked) ---
class ConceptCNN(nn.Module):
    def __init__(
        self,
        n_input_channels: int = CHANNELS,
        n_output: int = N_OUTPUT,
        n_hiddens: List[int] = CNN_CONFIG["n_hiddens"],
        n_layers: int = CNN_CONFIG["n_layers"],
    ):
        super().__init__()
        print(f"Initializing ConceptCNN: n_input_channels={n_input_channels}, n_output={n_output}, n_hiddens={n_hiddens}, n_layers={n_layers}")
        if not n_hiddens:
            raise ValueError("n_hiddens list cannot be empty for CNN")

        conv_layers = []
        current_channels = n_input_channels
        current_dim = 32 # Assuming 32x32 input images
        # Convolutional blocks
        for i, n_hidden in enumerate(n_hiddens):
            conv_layers.append(nn.Conv2d(current_channels, n_hidden, 3, 1, 1))
            conv_layers.append(nn.BatchNorm2d(n_hidden))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.AvgPool2d(2))
            current_channels = n_hidden
            current_dim = current_dim // 2

        self.conv_blocks = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Calculate flattened size
        n_penultimate = int(n_hiddens[-1] * current_dim * current_dim)

        # Fully connected layers
        fc_layers = []
        num_conv_blocks = len(n_hiddens)
        # Add dense layers based on the total n_layers parameter
        for i in range(max(0, n_layers - num_conv_blocks)):
            layer_in_dim = n_penultimate if i == 0 else n_penultimate # Or use a specific hidden dim? Using n_penultimate for now.
            layer_out_dim = n_penultimate
            fc_layers.append(nn.Linear(layer_in_dim, layer_out_dim))
            fc_layers.append(nn.BatchNorm1d(layer_out_dim))
            fc_layers.append(nn.ReLU())

        # Final output layer
        self.output_layer = nn.Linear(n_penultimate, n_output)

        # Store FC layers if any exist
        self.fc_blocks = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()

        # --- Hooking Attributes ---
        self.activations = {}
        self._hook_handles = []

    # --- Hooking Methods --- 
    def _get_activation(self, name):
        # Same as MLP's hook method
        def hook(model, input, output):
            act_input = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
            act_output = output.detach().clone()
            self.activations[name + '_input'] = act_input
            self.activations[name + '_output'] = act_output
        return hook

    def _register_hooks(self):
        self.remove_hooks()
        # Hook after first Conv block (after pooling)
        if len(self.conv_blocks) >= 4:
            self._hook_handles.append(self.conv_blocks[3].register_forward_hook(self._get_activation('conv_1')))
        # Hook after last Conv block (after pooling)
        if len(self.conv_blocks) > 0:
             self._hook_handles.append(self.conv_blocks[-1].register_forward_hook(self._get_activation('conv_last')))
        # Hook after flatten
        self._hook_handles.append(self.flatten.register_forward_hook(self._get_activation('flatten')))
        # Hook before final output layer (output of fc_blocks or flatten if no fc_blocks)
        if isinstance(self.fc_blocks, nn.Identity):
            # Input to output_layer is output of flatten
             self._hook_handles.append(self.flatten.register_forward_hook(self._get_activation('pre_output')))
        elif len(self.fc_blocks) > 0:
            # Input to output_layer is output of last ReLU in fc_blocks
             self._hook_handles.append(self.fc_blocks[-1].register_forward_hook(self._get_activation('pre_output')))
        # Hook final output
        self._hook_handles.append(self.output_layer.register_forward_hook(self._get_activation('output')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}
    # --- End Hooking Methods --- 

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.fc_blocks(x)
        x = self.output_layer(x)
        return x

# --- Transformer Model (Copied from ManyPaths/models.py and Hooked) ---
class ConceptTransformer(nn.Module):
    def __init__(
        self,
        n_input: int, # Features per patch or bit dimension
        n_output: int = N_OUTPUT,
        d_model: int = TRANSFORMER_CONFIG["d_model"],
        nhead: int = TRANSFORMER_CONFIG["nhead"],
        num_layers: int = TRANSFORMER_CONFIG["num_layers"],
        dim_feedforward: int = 2 * TRANSFORMER_CONFIG["d_model"],
    ):
        super().__init__()
        print(f"Initializing ConceptTransformer: n_input={n_input}, n_output={n_output}, d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        self.input_proj = nn.Linear(n_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model) # Assumes max_len=64 from PositionalEncoding default
        layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.decoder = nn.Linear(d_model, n_output)

        # --- Hooking Attributes ---
        self.activations = {}
        self.attention_weights = {}
        self._hook_handles = []

    # --- Hooking Methods --- 
    def _get_activation(self, name):
        # Same as MLP's hook method
        def hook(model, input, output):
            # TransformerEncoderLayer returns (output, attn_weights) tuple
            act_input = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
            if isinstance(output, tuple):
                 act_output = output[0].detach().clone()
            else:
                 act_output = output.detach().clone()
            self.activations[name + '_input'] = act_input
            self.activations[name + '_output'] = act_output
        return hook

    # Hook specifically for attention weights from TransformerEncoderLayer
    def _get_attention(self, name):
         def hook(model, input, output):
             if isinstance(output, tuple) and len(output) > 1:
                 attn_weights = output[1].detach().clone()
                 self.attention_weights[name + '_attention'] = attn_weights
         return hook

    def _register_hooks(self):
        self.remove_hooks()
        # Hook after input projection
        self._hook_handles.append(self.input_proj.register_forward_hook(self._get_activation('input_proj')))
        # Hook after positional encoding? Maybe hook input_proj output is enough.
        # Hook output of each encoder layer and its attention weights
        for i, layer in enumerate(self.encoder):
            # Hook activation output
            self._hook_handles.append(layer.register_forward_hook(self._get_activation(f'encoder_layer_{i}')))
            # Hook attention weights
            self._hook_handles.append(layer.register_forward_hook(self._get_attention(f'encoder_layer_{i}')))
        # Hook input to the final decoder/output layer
        # This is the output of the last encoder layer
        if len(self.encoder) > 0:
             last_encoder_layer = self.encoder[-1]
             # Need to hook the *output* of the last layer to get the input to the decoder
             # But the hook is on the layer itself. We capture its output via the standard activation hook.
             self._hook_handles.append(last_encoder_layer.register_forward_hook(self._get_activation('pre_output_temp')))
             # Note: the 'pre_output_temp_output' activation will be the input to the decoder

        # Hook final output
        self._hook_handles.append(self.decoder.register_forward_hook(self._get_activation('output')))

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.activations = {}
        self.attention_weights = {}
    # --- End Hooking Methods --- 

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        # Manually iterate to capture attention weights
        for i, layer in enumerate(self.encoder):
            x, _ = layer(x) # Discard attn weights here, hooks capture them
        # Use the output of the last token's representation for classification
        # Assumes sequence dimension is dim 1 (batch_first=True)
        final_token_representation = x[:, -1, :]
        output = self.decoder(final_token_representation)
        return output

# --- 3. Utility Functions ---
# Adapted from modular_arithmetic_maml.py

# --- Activation/Attention Extraction --- 
def get_all_activations(model, dataloader, device, layer_names):
    """Extracts activations from specified layers for all data in the dataloader.

    Assumes dataloader yields batches where each item is (support_x, support_y, query_x, query_y).
    Activations are extracted using the query_x set.
    """
    model.eval()
    model.to(device)
    if not hasattr(model, 'activations'):
        print(f"Warning: Model {type(model).__name__} does not have 'activations' attribute. Cannot extract.")
        return {}, None, None
    if hasattr(model, '_register_hooks'):
        model._register_hooks()
    else:
        print(f"Warning: Model {type(model).__name__} does not have '_register_hooks' method.")

    all_activations = {name: [] for name in layer_names}
    all_labels = []
    all_inputs = []

    desc = f"Extracting Activations ({type(model).__name__})"
    with torch.no_grad():
        # The dataloader here should be configured to yield batches of the *query* sets
        # from the specific task(s) we want to analyze.
        analysis_iterator = tqdm(dataloader, desc=desc, leave=False)
        for batch_data in analysis_iterator:
            # Assuming batch_data yields query_x, query_y directly
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                 inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            else:
                 print(f"Warning: Unexpected data format from dataloader in get_all_activations (expected X, y). Got {type(batch_data)}. Skipping batch.")
                 continue

            _ = model(inputs) # Forward pass to trigger hooks

            # Store activations
            for name in layer_names:
                 activation_data = None
                 if name in model.activations:
                     activation_data = model.activations[name].cpu().numpy()
                 elif name == 'pre_output' and 'pre_output_temp_output' in model.activations:
                     activation_data = model.activations['pre_output_temp_output'].cpu().numpy()
                 else:
                     print(f"Warning: Activation key '{name}' (or related) not found in model.activations.")
                     continue # Skip this layer if activation not found

                 # Process activation data (flatten, select token)
                 if activation_data is not None and activation_data.ndim > 2:
                     if 'encoder_layer' in name and isinstance(model, ConceptTransformer) and activation_data.ndim == 3:
                         # Use the representation of the last token (used for classification)
                         activation_data = activation_data[:, -1, :]
                     else:
                         # Flatten other multi-dim activations
                         activation_data = activation_data.reshape(activation_data.shape[0], -1)
                 all_activations[name].append(activation_data)

            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate results from all batches/tasks
    concatenated_activations = {}
    for name in layer_names:
        if all_activations[name]:
             # Ensure consistent shapes before concatenating
             shapes = [a.shape for a in all_activations[name]]
             if len(set(s[1:] for s in shapes)) > 1:
                  print(f"Warning: Inconsistent feature dimensions for layer '{name}'. Skipping concatenation.")
                  concatenated_activations[name] = None
                  continue
             try:
                 concatenated_activations[name] = np.concatenate(all_activations[name], axis=0)
             except ValueError as e:
                  print(f"Error concatenating activations for layer '{name}': {e}")
                  concatenated_activations[name] = None
        else:
             concatenated_activations[name] = None # Use None if no activations were collected
             # print(f"Warning: No activations collected for layer '{name}'.") # Reduce verbosity

    if hasattr(model, 'remove_hooks'):
        model.remove_hooks()

    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    all_inputs = np.concatenate(all_inputs, axis=0) if all_inputs else np.array([])

    # Ensure labels are 1D array for plotting/analysis (BCE loss expects N, 1)
    if all_labels.ndim > 1:
         # print(f"Squeezing labels from shape {all_labels.shape}")
         all_labels = all_labels.squeeze()
         if all_labels.ndim == 0 and len(all_inputs) == 1: # Handle case of single sample squeeze
             all_labels = np.array([all_labels.item()])
         elif all_labels.ndim != 1:
             print(f"Warning: Labels could not be squeezed to 1D. Final shape: {all_labels.shape}")


    return concatenated_activations, all_labels, all_inputs

# Remove get_all_attention_weights for vanilla script if not needed
# def get_all_attention_weights(...)

# --- Dimensionality Reduction Visualization --- 
def plot_dimensionality_reduction(
    activations, 
    labels, # Main concept labels (0/1)
    method='tsne', 
    n_components=2, 
    title='Activation Visualization', 
    color_label='Concept Label (0/1)', 
    perplexity=10, 
    n_iter=1000,
    feature_labels=None, # Optional: Pass the original N_FEATURES labels (e.g., shape [N, 4])
    color_by_feature_index=None, # Optional: Index (0-3) of the feature to color by
    feature_names=None # Optional: List like ['Color', 'Shape', 'Size', 'Style']
    ):
    """ Plots PCA or T-SNE of activations, colored by main labels or a specific feature."""
    import time
    import matplotlib.cm as cm # For distinct colors

    if activations is None or activations.size == 0:
        print(f"Skipping plot '{title}': No activation data.")
        return
    if labels is None or labels.size == 0:
        print(f"Skipping plot '{title}': No label data.")
        return

    # Ensure activations are 2D
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)

    num_samples = activations.shape[0]
    if num_samples == 0: print(f"Skipping plot '{title}': Zero samples."); return

    # Ensure labels are 1D and match sample count
    if labels.ndim > 1: labels = labels.squeeze()
    # Handle case where squeeze results in 0-dim array for single sample
    if labels.ndim == 0 and num_samples == 1:
        labels = np.array([labels.item()])
    elif labels.ndim != 1:
        print(f"Warning: Labels could not be squeezed to 1D for plotting '{title}'. Shape: {labels.shape}. Skipping.")
        return

    if labels.shape[0] != num_samples:
        print(f"Warning: Label count ({labels.shape[0]}) doesn't match activation count ({num_samples}) for '{title}'. Skipping plot.")
        return

    # Subsampling for large datasets (especially for T-SNE)
    # Concept dataset query set is small (16), so unlikely to be needed
    max_samples_plot = 5000
    if num_samples > max_samples_plot:
        print(f"Subsampling to {max_samples_plot} points for {method.upper()} plot: {title}")
        indices = np.random.choice(num_samples, max_samples_plot, replace=False)
        activations, labels = activations[indices], labels[indices]
        num_samples = max_samples_plot

    # Scale data
    scaler = StandardScaler()
    try:
        scaled_activations = scaler.fit_transform(activations)
        if np.any(np.isnan(scaled_activations)) or np.any(np.isinf(scaled_activations)):
            print(f"Warning: NaNs/Infs found in scaled activations for '{title}'. Using nan_to_num.")
            scaled_activations = np.nan_to_num(scaled_activations)
        if num_samples > 1 and np.all(np.var(scaled_activations, axis=0) < 1e-9):
             print(f"Warning: Near-zero variance in scaled activations for '{title}'. {method.upper()} might be unstable.")
    except ValueError as e:
        print(f"Error scaling activations for '{title}': {e}. Skipping plot.")
        return

    # Perform dimensionality reduction
    print(f"Running {method.upper()} on {num_samples} samples for '{title}'...")
    start_time = time.time() # Requires import time
    if method == 'tsne':
        # Adjust perplexity if num_samples is small (Concept has 16 query points)
        effective_perplexity = min(perplexity, max(1, num_samples - 2))
        if effective_perplexity != perplexity:
             print(f"  Adjusted T-SNE perplexity from {perplexity} to {effective_perplexity} due to sample size.")
        reducer = TSNE(n_components=n_components, random_state=SEED, perplexity=effective_perplexity, n_iter=max(250, n_iter), init='pca', learning_rate='auto')
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=SEED)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    try:
        reduced_activations = reducer.fit_transform(scaled_activations)
        print(f"  {method.upper()} finished in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during {method.upper()} for '{title}': {e}")
        return

    # Plotting
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels))
    # Use a simple map for binary labels (e.g., coolwarm, RdYlBu)
    cmap = plt.get_cmap('coolwarm', 2) if len(unique_labels) <= 2 else plt.get_cmap('viridis', len(unique_labels))

    if n_components == 2:
        scatter = plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c=labels, cmap=cmap, alpha=0.8, s=40)
        plt.xlabel(f"{method.upper()} Component 1"); plt.ylabel(f"{method.upper()} Component 2")
    elif n_components == 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10,8)) # Create figure explicitly for 3D
            ax = fig.add_subplot(projection='3d')
            scatter = ax.scatter(reduced_activations[:, 0], reduced_activations[:, 1], reduced_activations[:, 2], c=labels, cmap=cmap, alpha=0.8, s=40)
            ax.set_xlabel(f"{method.upper()} Comp 1"); ax.set_ylabel(f"{method.upper()} Comp 2"); ax.set_zlabel(f"{method.upper()} Comp 3")
        except ImportError:
            print("Could not import Axes3D for 3D plot. Skipping 3D plot for '{title}'.")
            plt.close() # Close the empty figure
            return
        except Exception as e:
             print(f"Error creating 3D plot for '{title}': {e}. Skipping plot.")
             plt.close(); return
    else:
        raise ValueError("n_components must be 2 or 3")

    plt.title(f"{title} ({method.upper()})", fontsize=12)
    # Add colorbar based on the labels used for plotting
    if len(unique_labels) >= 2:
        try:
            # Adjust ticks based on whether it's main label or feature
            ticks = [0, 1] if len(unique_labels) == 2 else unique_labels
            cbar = plt.colorbar(scatter, ticks=ticks)
            cbar.set_label(color_label)
            if len(ticks) == 2: # Make sure binary labels are clear
                 cbar.set_ticklabels(['0', '1'])
        except Exception as e:
            print(f"Could not add colorbar for '{title}': {e}")

    plt.show()

def plot_weight_histograms(model, model_name="Model"):
    """ Plots histograms of weights for key layers in the model. """
    print(f"\n--- Plotting Weight Histograms for {model_name} ---")
    # Identify key layers with weights (Linear, Conv2d)
    layers_to_plot = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Try to get a meaningful name
            simple_name = name.split('.')[-1] # Often the last part is descriptive
            # Avoid plotting hooks internal layers if names clash
            if simple_name.startswith('_'): continue
            layers_to_plot.append((f"{name} ({type(module).__name__})", module))

    if not layers_to_plot:
        print("No Linear or Conv2d layers found to plot histograms for.")
        return

    num_layers = len(layers_to_plot)
    # Adjust layout based on number of layers
    ncols = min(4, num_layers)
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
    axes = axes.flatten()

    fig.suptitle(f'{model_name} - Weight Histograms', fontsize=14)

    for i, (layer_name, layer) in enumerate(layers_to_plot):
        if hasattr(layer, 'weight') and layer.weight is not None:
            weights = layer.weight.data.cpu().numpy().flatten()
            axes[i].hist(weights, bins=50, color='skyblue', edgecolor='black')
            axes[i].set_title(layer_name, fontsize=9)
            axes[i].set_xlabel("Weight Value", fontsize=8)
            axes[i].set_ylabel("Frequency", fontsize=8)
            axes[i].tick_params(axis='both', which='major', labelsize=7)
            # Add mean/std text
            mean_w = np.mean(weights)
            std_w = np.std(weights)
            axes[i].text(0.95, 0.95, f'μ={mean_w:.2f}\nσ={std_w:.2f}',
                         ha='right', va='top', transform=axes[i].transAxes, fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
        else:
             axes[i].set_title(f'{layer_name}\n(No weight attr.)', fontsize=9)
             axes[i].axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def visualize_cnn_filters(model, model_name="CNN"):
    """ Visualizes the filters of the first Conv2d layer of a CNN. """
    print(f"\n--- Visualizing First Conv Layer Filters for {model_name} ---")
    first_conv_layer = None
    layer_name = "N/A"

    # Find the first Conv2d layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv_layer = module
            layer_name = name
            break

    if first_conv_layer is None:
        print("No Conv2d layer found in the model.")
        return

    if not hasattr(first_conv_layer, 'weight'):
        print(f"Layer '{layer_name}' has no 'weight' attribute.")
        return

    weights = first_conv_layer.weight.data.cpu()
    # Shape: (out_channels, in_channels, kernel_height, kernel_width)
    out_channels, in_channels, kh, kw = weights.shape
    print(f"Found layer '{layer_name}' with {out_channels} filters of size ({in_channels}, {kh}, {kw}).")

    # Normalize weights for visualization (per filter)
    min_vals = weights.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
    max_vals = weights.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
    weights_norm = (weights - min_vals) / (max_vals - min_vals + 1e-6)

    # Determine grid size
    ncols = min(8, out_channels) # Show max 8 filters per row
    nrows = (out_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f'{model_name} - First Conv Layer ({layer_name}) Filters ({out_channels} total)', fontsize=12)

    for i in range(out_channels):
        filt = weights_norm[i]
        # If input has multiple channels (e.g., 3 for RGB), show them side-by-side or average?
        # For simplicity, let's show the average across input channels if C > 1
        if in_channels > 1:
            filt_display = filt.mean(dim=0) # Average across input channels
            cmap = 'gray' # Display average as grayscale
        else:
            filt_display = filt.squeeze(0) # Remove input channel dim if C=1
            cmap = 'viridis'

        ax = axes[i]
        im = ax.imshow(filt_display.numpy(), cmap=cmap, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Filter {i}', fontsize=8)

    # Hide unused subplots
    for j in range(out_channels, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pca_layer_weights(layer, layer_name="Layer", model_name="Model"):
    """ Performs PCA on the weight vectors of neurons/filters in a layer and plots the first 2 components. """
    print(f"\n--- Plotting PCA of Weights for {model_name} - {layer_name} ---")
    if not hasattr(layer, 'weight') or layer.weight is None:
        print(f"Layer '{layer_name}' has no weights to visualize.")
        return

    weights = layer.weight.data.cpu().numpy()
    original_shape = weights.shape
    n_output_units = original_shape[0] # Neurons (Linear) or Filters (Conv)

    if n_output_units <= 2:
        print(f"Skipping PCA for layer '{layer_name}': Number of output units ({n_output_units}) is too small.")
        return

    # Reshape weights: (n_output_units, n_input_features_per_unit)
    if weights.ndim > 2: # e.g., Conv layers
        weights_reshaped = weights.reshape(n_output_units, -1)
    elif weights.ndim == 2: # Linear layers
        weights_reshaped = weights
    else:
        print(f"Skipping PCA for layer '{layer_name}': Unexpected weight dimension {weights.ndim}.")
        return

    n_features = weights_reshaped.shape[1]
    if n_features <= 1:
        print(f"Skipping PCA for layer '{layer_name}': Number of features per unit ({n_features}) is too small.")
        return

    print(f"Running PCA on {n_output_units} weight vectors (each {n_features} dims) for layer '{layer_name}'...")

    # Scale features before PCA
    try:
        scaler = StandardScaler()
        weights_scaled = scaler.fit_transform(weights_reshaped)
    except ValueError as e:
         print(f"  Error scaling weights for PCA: {e}. Skipping layer.")
         return

    # Apply PCA
    n_components = 2
    try:
        pca = PCA(n_components=n_components, random_state=SEED)
        weights_pca = pca.fit_transform(weights_scaled)
        explained_variance = pca.explained_variance_ratio_
        print(f"  PCA finished. Explained variance: {explained_variance.sum()*100:.2f}% ({explained_variance[0]*100:.2f}%, {explained_variance[1]*100:.2f}%)")
    except Exception as e:
         print(f"  Error during PCA: {e}. Skipping layer.")
         return

    # Plot PCA results
    plt.figure(figsize=(8, 7))
    cmap = plt.get_cmap('viridis', n_output_units)
    scatter = plt.scatter(weights_pca[:, 0], weights_pca[:, 1], c=np.arange(n_output_units), cmap=cmap, alpha=0.8, s=50)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}%)")
    plt.title(f"PCA of Weights: {model_name} - {layer_name}\n(Points are Neurons/Filters, Colored by Index)")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add colorbar
    try:
        cbar = plt.colorbar(scatter, ticks=np.linspace(0, n_output_units-1, min(10, n_output_units), dtype=int))
        cbar.set_label("Neuron / Filter Index")
    except Exception as e:
        print(f"Could not add colorbar: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Linear Probing Functions ---
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd # For structuring heatmap data

def train_probe(activations, labels, test_size=0.3, random_state=SEED):
    """Trains a logistic regression probe on activations to predict labels.

    Returns the test accuracy (%). Handles potential issues like few samples or classes.
    """
    if activations is None or activations.size == 0:
        print("  Probe Error: No activation data.")
        return 0.0
    if labels is None or labels.size == 0:
        print("  Probe Error: No label data.")
        return 0.0

    # Ensure activations are 2D
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)

    if activations.shape[0] != labels.shape[0]:
         print(f"  Probe Error: Activation/label count mismatch ({activations.shape[0]} vs {labels.shape[0]}).")
         return 0.0
    if activations.shape[0] < 5: # Need enough samples for train/test split
         print(f"  Probe Warning: Too few samples ({activations.shape[0]}) for reliable probing. Returning 0.")
         return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # print(f"  Probe Info: Only one class ({unique_labels}) present. Cannot train probe.")
        # Return 100% if only one class and all predictions are that class, otherwise 0? Or just 0.
        return 0.0

    try:
        # Stratified split if possible
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        # Fallback if stratification fails (e.g., too few samples in one class)
        # print("  Probe Warning: Stratification failed. Using non-stratified split.")
        try:
             X_train, X_test, y_train, y_test = train_test_split(
                 activations, labels, test_size=test_size, random_state=random_state
             )
        except Exception as e:
            print(f"  Probe Error: train_test_split failed: {e}")
            return 0.0

    # Scale data
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Handle potential NaNs/Infs after scaling (e.g., zero variance columns)
        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
             # print("  Probe Warning: NaNs/Infs in scaled training data. Using nan_to_num.")
             X_train_scaled = np.nan_to_num(X_train_scaled)
        if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
             # print("  Probe Warning: NaNs/Infs in scaled test data. Using nan_to_num.")
             X_test_scaled = np.nan_to_num(X_test_scaled)
    except ValueError as e:
        print(f"  Probe Error: Scaling failed: {e}")
        return 0.0

    # Train Logistic Regression probe
    # Use balanced class weight for potentially imbalanced features
    probe = LogisticRegression(random_state=random_state, max_iter=500, C=0.1, solver='liblinear', class_weight='balanced')
    try:
        probe.fit(X_train_scaled, y_train)
        accuracy = probe.score(X_test_scaled, y_test)
        return accuracy * 100
    except ValueError as e:
         # Catch errors like "This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0"
         print(f"  Probe Error: Fitting/scoring failed: {e}")
         return 0.0
    except Exception as e:
        print(f"  Probe Error: Fitting/scoring failed unexpectedly: {e}")
        return 0.0

def plot_probing_heatmap(probing_results, feature_names, title="Linear Probing Accuracy (%)"):
    """ Creates a heatmap visualizing probing accuracy across layers and features for all models. """
    if not probing_results:
        print("No probing results to plot.")
        return

    # Structure: results[model_name][layer_name][feature_name] = accuracy
    all_data = []
    model_names = list(probing_results.keys())
    # Get unique layer names across all models (maintain some order if possible)
    layer_names_ordered = []
    seen_layers = set()
    for model_name in model_names:
         for layer_name in probing_results[model_name].keys():
              if layer_name not in seen_layers:
                   layer_names_ordered.append(layer_name)
                   seen_layers.add(layer_name)

    for model_name in model_names:
        for layer_name in layer_names_ordered:
            for i, feature_name in enumerate(feature_names):
                accuracy = probing_results.get(model_name, {}).get(layer_name, {}).get(feature_name, np.nan)
                all_data.append({
                    "Model": model_name,
                    "Layer": layer_name,
                    "Feature": feature_name,
                    "Accuracy": accuracy
                })

    if not all_data:
         print("No valid data points found in probing results.")
         return

    df = pd.DataFrame(all_data)

    # Create FacetGrid: one plot per Model
    g = sns.FacetGrid(df, col="Model", col_wrap=min(len(model_names), 3), height=max(4, len(layer_names_ordered)*0.5), aspect=0.8)

    # Map heatmap to each facet
    g.map_dataframe(lambda data, color: sns.heatmap(data.pivot(index="Layer", columns="Feature", values="Accuracy"), 
                                                  annot=True, fmt=".1f", linewidths=.5, cmap="viridis", vmin=0, vmax=100, cbar=False))

    g.fig.suptitle(title, y=1.02, fontsize=14)
    g.set_titles(col_template="{col_name}")
    # Adjust tick labels
    for ax in g.axes.flat:
         ax.tick_params(axis='y', labelrotation=0, labelsize=8)
         ax.tick_params(axis='x', labelrotation=45, labelsize=8)

    # Add a single color bar
    # cbar_ax = g.fig.add_axes([1.01, .3, .02, .4]) # Adjust position as needed
    # norm = plt.Normalize(0, 100)
    # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    # sm.set_array([])
    # g.fig.colorbar(sm, cax=cbar_ax, label="Accuracy (%)")

    g.fig.tight_layout()
    plt.show()

# --- Standard Training Function (Adapted from modular_arithmetic_vanilla.py) ---
def train_model_standard(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="Model", model_save_dir="saved_models"):
    """ Trains a model using standard SGD/Adam with early stopping based on validation loss/accuracy. """
    print(f"\n--- Training {model_name} (Standard) ---")
    model.to(device)

    # Early Stopping Parameters
    early_stopping_patience = 10
    early_stopping_threshold_acc = 1.0 # Min % acc improvement to reset patience
    epochs_no_improve = 0
    best_val_metric = -1.0 # Using accuracy, higher is better
    best_model_path = os.path.join(model_save_dir, f"{model_name}_standard_concept_best.pth")

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
            loss = criterion(outputs, labels) # BCEWithLogitsLoss
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            # Calculate accuracy for binary classification with logits
            predicted = torch.sigmoid(outputs) > 0.5
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_iterator.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0

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
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = torch.sigmoid(outputs) > 0.5
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Early Stopping Check (based on validation accuracy)
        current_metric = val_acc
        if current_metric > best_val_metric + early_stopping_threshold_acc:
            print(f"  Validation accuracy improved significantly ({best_val_metric:.2f}% -> {current_metric:.2f}%). Saving model.")
            best_val_metric = current_metric
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        elif current_metric > best_val_metric:
             print(f"  Validation accuracy improved slightly ({best_val_metric:.2f}% -> {current_metric:.2f}%). Saving model.")
             best_val_metric = current_metric # Update best even for small improvements
             torch.save(model.state_dict(), best_model_path)
             epochs_no_improve += 1 # Still count towards patience unless significant
        else:
            epochs_no_improve += 1
            print(f"  Validation accuracy did not improve ({current_metric:.2f}% vs best {best_val_metric:.2f}%). Patience: {epochs_no_improve}/{early_stopping_patience}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    print(f"Finished Standard Training {model_name}. Best Val Acc: {best_val_metric:.2f}%")
    if os.path.exists(best_model_path):
        print(f"Loading best standard model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Warning: Best standard model file not found at {best_model_path}.")
    return model


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    print("--- Concept Learning Vanilla Training & Visualization Script ---")

    # --- 1. Define a Single Concept Task ---
    print("\n--- Defining Single Concept Task ---")
    # Generate a fixed hypothesis (concept rule) for this run
    # Use the placeholder DNFHypothesis for now
    concept_hyp = DNFHypothesis(n_features=N_FEATURES)
    all_feature_vectors = torch.tensor(FEATURE_VALUES, dtype=torch.float)
    all_labels = torch.tensor([concept_hyp.function(f) for f in FEATURE_VALUES], dtype=torch.float).unsqueeze(1)

    # Prepare data based on DATA_TYPE and model
    # We need to generate/process the 16 inputs (images or bits)
    if DATA_TYPE == "image":
        print("Generating and preprocessing concept images...")
        all_images_np = [generate_concept(bits, scale=1.0).transpose(2, 0, 1) for bits in FEATURE_VALUES]
        X_images_all = torch.tensor(np.array(all_images_np), dtype=torch.float)
        # Normalize images
        mean = X_images_all.mean(dim=[0, 2, 3])
        std = X_images_all.std(dim=[0, 2, 3])
        std[std == 0] = 1.0
        transform = transforms.Normalize(mean=mean, std=std)
        X_images_norm = transform(X_images_all)
        # Note: Further processing (flatten, patching) happens inside model init/forward
        all_inputs_for_mlp = X_images_norm.reshape(16, -1)
        # Need patch function if using transformer
        patch_fn = BaseMetaDataset()._image_to_patches # Instantiate base class to access method
        all_inputs_for_tf = patch_fn(X_images_norm) # Default patch size 4
        all_inputs_for_cnn = X_images_norm

    elif DATA_TYPE == "bits":
        print("Using bit representations...")
        X_bits = all_feature_vectors * 2.0 - 1.0 # Convert 0/1 to -1/1
        all_inputs_for_mlp = X_bits
        all_inputs_for_cnn = X_bits # CNN might not be ideal for bits
        all_inputs_for_tf = X_bits.unsqueeze(-1) # Add feature dim for transformer
    else:
        raise ValueError(f"Unsupported DATA_TYPE: {DATA_TYPE}")

    # --- 2. Create Standard Dataset & DataLoaders ---
    print("\n--- Creating Training/Validation DataLoaders ---")
    num_examples = len(all_labels)
    indices = list(range(num_examples))
    split_idx = int(np.floor(VALIDATION_SPLIT * num_examples))
    # Ensure reproducibility of split
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    # Create TensorDatasets for each model type using the correct input format
    # We need separate datasets because the input tensor shape varies
    datasets = {}
    if DATA_TYPE == "image":
        datasets['mlp'] = (torch.utils.data.TensorDataset(all_inputs_for_mlp[train_indices], all_labels[train_indices]),
                             torch.utils.data.TensorDataset(all_inputs_for_mlp[val_indices], all_labels[val_indices]))
        datasets['cnn'] = (torch.utils.data.TensorDataset(all_inputs_for_cnn[train_indices], all_labels[train_indices]),
                             torch.utils.data.TensorDataset(all_inputs_for_cnn[val_indices], all_labels[val_indices]))
        datasets['transformer'] = (torch.utils.data.TensorDataset(all_inputs_for_tf[train_indices], all_labels[train_indices]),
                                  torch.utils.data.TensorDataset(all_inputs_for_tf[val_indices], all_labels[val_indices]))
    elif DATA_TYPE == "bits":
         datasets['mlp'] = (torch.utils.data.TensorDataset(all_inputs_for_mlp[train_indices], all_labels[train_indices]),
                              torch.utils.data.TensorDataset(all_inputs_for_mlp[val_indices], all_labels[val_indices]))
         datasets['cnn'] = datasets['mlp'] # Use MLP data for CNN with bits
         datasets['transformer'] = (torch.utils.data.TensorDataset(all_inputs_for_tf[train_indices], all_labels[train_indices]),
                                   torch.utils.data.TensorDataset(all_inputs_for_tf[val_indices], all_labels[val_indices]))

    dataloaders = {}
    for model_type, (train_ds, val_ds) in datasets.items():
        train_loader = DataLoader(train_ds, batch_size=STANDARD_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=STANDARD_BATCH_SIZE, shuffle=False)
        dataloaders[model_type] = (train_loader, val_loader)
        print(f"  {model_type.upper()}: Train size={len(train_ds)}, Val size={len(val_ds)}")

    # --- 3. Initialize Models ---
    print("\n--- Initializing Models ---")
    # Determine n_input based on DATA_TYPE and model
    if DATA_TYPE == "image":
        n_input_mlp = 3 * 32 * 32
        n_input_cnn = CHANNELS # CNN takes channels
        n_input_tf = CHANNELS * 4 * 4 # Features per patch (default patch size 4)
    elif DATA_TYPE == "bits":
        n_input_mlp = N_FEATURES
        n_input_cnn = N_FEATURES # Treat as flattened for CNN?
        n_input_tf = 1 # Feature dim is 1 for bits

    mlp_model = ConceptMLP(n_input=n_input_mlp, n_input_channels=CHANNELS, **MLP_CONFIG).to(DEVICE)
    cnn_model = ConceptCNN(n_input_channels=CHANNELS, **CNN_CONFIG).to(DEVICE) # CNN uses channels directly
    transformer_model = ConceptTransformer(n_input=n_input_tf, **TRANSFORMER_CONFIG).to(DEVICE)

    models = {"MLP": mlp_model, "CNN": cnn_model, "Transformer": transformer_model}

    # --- 4. Train Models ---
    criterion = nn.BCEWithLogitsLoss() # Binary classification
    trained_models = {}

    for model_name, model in models.items():
        print(f"\n--- Preparing Training for {model_name} --- ")
        optimizer = optim.Adam(model.parameters(), lr=STANDARD_LEARNING_RATE)
        train_loader, val_loader = dataloaders[model_name.lower()] # Get the correct dataloader
        trained_model = train_model_standard(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=STANDARD_EPOCHS, device=DEVICE, model_name=model_name,
            model_save_dir=MODEL_SAVE_DIR
        )
        trained_models[model_name] = trained_model

    # --- 5. Extract Activations & Visualize ---
    print("\n--- Activation Extraction & Visualization ---")
    # Create a DataLoader for the *full* dataset (all 16 examples) for visualization
    if DATA_TYPE == "image":
        full_dataset_mlp = torch.utils.data.TensorDataset(all_inputs_for_mlp, all_labels)
        full_dataset_cnn = torch.utils.data.TensorDataset(all_inputs_for_cnn, all_labels)
        full_dataset_tf = torch.utils.data.TensorDataset(all_inputs_for_tf, all_labels)
    elif DATA_TYPE == "bits":
        full_dataset_mlp = torch.utils.data.TensorDataset(all_inputs_for_mlp, all_labels)
        full_dataset_cnn = full_dataset_mlp
        full_dataset_tf = torch.utils.data.TensorDataset(all_inputs_for_tf, all_labels)

    full_dataloaders = {
        "MLP": DataLoader(full_dataset_mlp, batch_size=16, shuffle=False),
        "CNN": DataLoader(full_dataset_cnn, batch_size=16, shuffle=False),
        "Transformer": DataLoader(full_dataset_tf, batch_size=16, shuffle=False)
    }

    # Define layers to visualize (match hook names)
    mlp_layers_vis = ['input_layer_output', 'hidden_1_output', 'pre_output_output', 'output_output']
    cnn_layers_vis = ['conv_1_output', 'conv_last_output', 'flatten_output', 'pre_output_output', 'output_output']
    tf_layers_vis = ['input_proj_output'] + [f'encoder_layer_{i}_output' for i in range(TRANSFORMER_CONFIG["num_layers"])] + ['pre_output_temp_output', 'output_output']
    layer_map = {
        "MLP": mlp_layers_vis,
        "CNN": cnn_layers_vis,
        "Transformer": tf_layers_vis
        }
    # Define the specific key we want to plot (must be one of the keys listed above for the respective model)
    # final_layer_name = 'output_output' # Cannot visualize 1D output layer
    # Let's target the pre-output layer instead
    pre_output_keys = {
        "MLP": "pre_output_output",
        "CNN": "pre_output_output",
        "Transformer": "pre_output_temp_output" # The key stored by the hook
    }

    for model_name, model in trained_models.items():
        print(f"\n--- Visualizing Activations for {model_name} ---")
        dataloader = full_dataloaders[model_name]
        layers_to_extract = layer_map[model_name]

        activations, labels, _ = get_all_activations(model, dataloader, DEVICE, layers_to_extract)

        if not activations:
             print(f"No activations extracted for {model_name}. Skipping visualization.")
             continue

        print(f"Extracted layers: {list(activations.keys())}")

        # Define feature names for plots
        feature_names = ['Color', 'Shape', 'Size', 'Style']
        # Get the original feature vectors (0/1) used for generation
        # Ensure these align with the order of 'activations' and 'labels'
        original_feature_vectors = all_feature_vectors.numpy() # Shape (16, 4)

        # Iterate through all extracted layers for visualization
        for layer_key, layer_activations in activations.items():
             if layer_activations is None:
                 print(f"Skipping visualization for layer '{layer_key}': No data extracted.")
                 continue

             print(f"  Visualizing layer: {layer_key} (Shape: {layer_activations.shape})")

             # Check if the layer has more than 1 feature dimension
             if layer_activations.ndim < 2 or layer_activations.shape[1] <= 1:
                 print(f"  Skipping T-SNE/PCA for layer '{layer_key}': Layer dimension ({layer_activations.shape[1] if layer_activations.ndim >= 2 else 'N/A'}) is not > 1.")
                 continue

             # Plot 1: Colored by main concept label (Keep this one)
             print(f"    Plotting T-SNE colored by Concept Label...")
             plot_dimensionality_reduction(
                 layer_activations,
                 labels, # Main concept labels (0/1)
                 method='tsne',
                 title=f'{model_name} - Layer: {layer_key} ({DATA_TYPE})',
                 color_label='Concept Label (0/1)'
             )

             # Plots 2-5: Colored by individual features -> REMOVE THIS BLOCK
             # for feature_idx in range(N_FEATURES):
             #     print(f"    Plotting colored by Feature {feature_idx} ({feature_names[feature_idx]})...")
             #     plot_dimensionality_reduction(
             #         layer_activations,
             #         labels, # Pass main labels anyway (not used for coloring here)
             #         method='tsne',
             #         title=f'{model_name} - Layer: {layer_key} ({DATA_TYPE})', # Title will be modified inside
             #         feature_labels=original_feature_vectors,
             #         color_by_feature_index=feature_idx,
             #         feature_names=feature_names
             #     )

    # --- 6. Visualize Weights ---
    print("\n--- Weight Visualization ---")
    all_probing_results = {} # Initialize dictionary to store probing results

    for model_name, model in trained_models.items():
        print(f"\n--- Visualizing Activations & Probing for {model_name} ---")
        dataloader = full_dataloaders[model_name]
        layers_to_extract = layer_map[model_name]

        activations, labels, _ = get_all_activations(model, dataloader, DEVICE, layers_to_extract)

        if not activations:
             print(f"No activations extracted for {model_name}. Skipping visualization and probing.")
             continue

        print(f"Extracted layers: {list(activations.keys())}")

        # Define feature names for plots and probes
        feature_names = ['Color', 'Shape', 'Size', 'Style']
        # Get the original feature vectors (0/1) used for generation
        original_feature_vectors = all_feature_vectors.numpy() # Shape (16, 4)

        # Initialize probing results for this model
        all_probing_results[model_name] = {}

        # Iterate through all extracted layers for visualization and probing
        for layer_key, layer_activations in activations.items():
            all_probing_results[model_name][layer_key] = {} # Init results for this layer

            if layer_activations is None:
                print(f"Skipping visualization & probing for layer '{layer_key}': No data extracted.")
                continue

            print(f"\n  Analyzing layer: {layer_key} (Shape: {layer_activations.shape})")

            # Check if the layer has more than 1 feature dimension for visualization
            if layer_activations.ndim < 2 or layer_activations.shape[1] <= 1:
                print(f"    Skipping T-SNE/PCA for layer '{layer_key}': Layer dimension ({layer_activations.shape[1] if layer_activations.ndim >= 2 else 'N/A'}) is not > 1.")
            else:
                # Plot 1: Colored by main concept label
                print(f"    Plotting T-SNE colored by Concept Label...")
                plot_dimensionality_reduction(
                    layer_activations,
                    labels, # Main concept labels (0/1)
                    method='tsne',
                    title=f'{model_name} - Layer: {layer_key} ({DATA_TYPE})',
                    color_label='Concept Label (0/1)'
                )

            # --- Linear Probing ---
            print(f"    Running Linear Probes...")
            # Probe for each individual feature
            for i, feature_name in enumerate(feature_names):
                 print(f"      Probing for Feature: {feature_name}")
                 # Ensure feature labels are integer type if needed by probe
                 feature_labels_for_probe = original_feature_vectors[:, i].astype(int)
                 probe_accuracy = train_probe(layer_activations, feature_labels_for_probe)
                 all_probing_results[model_name][layer_key][feature_name] = probe_accuracy
                 print(f"        Accuracy: {probe_accuracy:.2f}%")

            # Also probe for the main concept label
            print(f"      Probing for Feature: Concept Label")
            concept_labels_for_probe = labels.astype(int) # Labels should already be 0/1
            probe_accuracy_concept = train_probe(layer_activations, concept_labels_for_probe)
            all_probing_results[model_name][layer_key]['Concept'] = probe_accuracy_concept # Use 'Concept' as the key
            print(f"        Accuracy: {probe_accuracy_concept:.2f}%")

        # --- Weight Visualization (Moved inside model loop for clarity) ---
        print(f"\n  Visualizing Weights for {model_name}")
        # 1. Weight Histograms
        plot_weight_histograms(model, model_name=model_name)

        # 2. CNN First Layer Filter Visualization
        if isinstance(model, ConceptCNN):
            visualize_cnn_filters(model, model_name=model_name)

        # 3. PCA of Layer Weights
        layers_to_pca = {}
        if isinstance(model, ConceptMLP):
            layers_to_pca[f"MLP Input Layer ({model.input_layer.out_features} units)"] = model.input_layer
            layers_to_pca[f"MLP Output Layer ({model.output_layer.out_features} units)"] = model.output_layer
        elif isinstance(model, ConceptCNN):
            first_conv_layer_tuple = next(((name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Conv2d)), (None, None))
            if first_conv_layer_tuple[1] is not None:
                 layers_to_pca[f"CNN First Conv ({first_conv_layer_tuple[1].out_channels} filters)"] = first_conv_layer_tuple[1]
            layers_to_pca[f"CNN Output Layer ({model.output_layer.out_features} units)"] = model.output_layer
        elif isinstance(model, ConceptTransformer):
             layers_to_pca[f"Transformer Input Proj ({model.input_proj.out_features} units)"] = model.input_proj
             if len(model.encoder) > 0:
                  first_encoder_layer = model.encoder[0]
                  layers_to_pca[f"Transformer Enc0 Attn ({first_encoder_layer.self_attn.embed_dim} units)"] = first_encoder_layer.self_attn
                  layers_to_pca[f"Transformer Enc0 FFN1 ({first_encoder_layer.linear1.out_features} units)"] = first_encoder_layer.linear1
             layers_to_pca[f"Transformer Decoder ({model.decoder.out_features} units)"] = model.decoder

        for layer_desc, layer_module in layers_to_pca.items():
             plot_pca_layer_weights(layer_module, layer_name=layer_desc, model_name=model_name)

    # --- Final Probing Visualization (Moved after all models are processed) ---
    print("\n--- Generating Probing Accuracy Heatmap ---")
    if all_probing_results:
        plot_probing_heatmap(all_probing_results, feature_names, title="Linear Probing Accuracy by Layer and Feature")
    else:
        print("Skipping heatmap generation: No probing results available.")


    print("\n--- Concept Learning Vanilla Script Finished ---")