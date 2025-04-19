import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import itertools

# --- Configuration ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Maze Configuration
MAZE_HEIGHT = 7
MAZE_WIDTH = 7
WALL = 0
PATH = 1
START = 2
GOAL = 3
CURRENT_POS_MARKER = 0.5 # Value to mark current position for CNN input

# Action Mapping
ACTIONS = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
ACTION_VECTORS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
NUM_ACTIONS = len(ACTIONS)

# Model Configuration (Examples)
HIDDEN_DIM = 64
LSTM_LAYERS = 1
TRANSFORMER_LAYERS = 2
N_HEAD = 4
MAX_SEQ_LEN = MAZE_HEIGHT * MAZE_WIDTH # Max possible path length

# Training Configuration
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Maze Definitions and Paths ---

MAZES = {}
PATHS = {}

# Define Mazes (Example: U-shape)
maze_u = np.ones((MAZE_HEIGHT, MAZE_WIDTH)) * WALL
maze_u[1, 1:6] = PATH
maze_u[2:6, 1] = PATH
maze_u[2:6, 5] = PATH
maze_u[1, 1] = START
maze_u[5, 5] = GOAL # Goal at the bottom right leg
MAZES['U'] = maze_u
# Define path for U-shape (traverse the U)
PATHS['U'] = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]

# Define Mazes (Example: T-shape)
maze_t = np.ones((MAZE_HEIGHT, MAZE_WIDTH)) * WALL
maze_t[1, 1:6] = PATH # Top bar
maze_t[2:6, 3] = PATH # Stem
maze_t[5, 3] = START # Start at bottom of stem
maze_t[1, 5] = GOAL   # Goal at end of right arm
MAZES['T'] = maze_t
PATHS['T'] = [(5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (1, 4), (1, 5)]

# Define Mazes (Example: M-shape - simplified)
maze_m = np.ones((MAZE_HEIGHT, MAZE_WIDTH)) * WALL
maze_m[1:6, 1] = PATH
maze_m[1:4, 3] = PATH
maze_m[3:6, 3] = PATH
maze_m[1:6, 5] = PATH
maze_m[1, 1] = START
maze_m[5, 5] = GOAL # Goal at bottom right
MAZES['M'] = maze_m
PATHS['M'] = [(1,1), (2,1), (3,1), (4,1), (5,1), (5,2), (5,3), (4,3), (3,3), (2,3), (1,3), (1,4), (1,5), (2,5), (3,5), (4,5), (5,5)] # Longer path

# Define Mazes (Example: S-shape)
maze_s = np.ones((MAZE_HEIGHT, MAZE_WIDTH)) * WALL
maze_s[1, 1:6] = PATH # Top segment
maze_s[2:4, 1] = PATH # Upper vertical
maze_s[3, 1:6] = PATH # Middle segment
maze_s[4:6, 5] = PATH # Lower vertical
maze_s[5, 1:6] = PATH # Bottom segment
maze_s[1, 1] = START
maze_s[5, 1] = GOAL
MAZES['S'] = maze_s
PATHS['S'] = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1), (4, 1), (5, 1)]

# Define Mazes (Example: H-shape)
maze_h = np.ones((MAZE_HEIGHT, MAZE_WIDTH)) * WALL
maze_h[1:6, 1] = PATH # Left vertical
maze_h[1:6, 5] = PATH # Right vertical
maze_h[3, 1:6] = PATH # Horizontal bar
maze_h[1, 1] = START
maze_h[5, 5] = GOAL
MAZES['H'] = maze_h
PATHS['H'] = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5)]

MAZE_IDS = {name: i for i, name in enumerate(MAZES.keys())}
NUM_MAZES = len(MAZES)

# --- Dataset --- 

class MazeNavigationDataset(Dataset):
    """Generates state-action pairs for navigating predefined maze paths."""
    def __init__(self, maze_dict, path_dict, model_type):
        self.maze_dict = maze_dict
        self.path_dict = path_dict
        self.model_type = model_type.lower()
        self.data = self._generate_data()

    def _get_action(self, pos1, pos2):
        """Determines the action needed to move from pos1 to pos2."""
        delta_r = pos2[0] - pos1[0]
        delta_c = pos2[1] - pos1[1]
        for action, (dr, dc) in ACTION_VECTORS.items():
            if dr == delta_r and dc == delta_c:
                return ACTIONS[action]
        return -1 # Should not happen for adjacent steps in path

    def _generate_data(self):
        all_inputs = []
        all_actions = []
        all_positions = [] # Store (r, c) for visualization coloring
        all_maze_ids = []  # Store maze ID

        for maze_name, path in self.path_dict.items():
            maze_id = MAZE_IDS[maze_name]
            maze_grid = self.maze_dict[maze_name]

            current_path_coords = [] # For LSTM/Transformer sequence

            for i in range(len(path) - 1):
                current_pos = path[i]
                next_pos = path[i+1]
                action_idx = self._get_action(current_pos, next_pos)

                if action_idx == -1:
                    print(f"Warning: Invalid step in path {maze_name}: {current_pos} -> {next_pos}")
                    continue

                current_path_coords.append(torch.tensor(current_pos, dtype=torch.float32))

                # Format input based on model type
                if self.model_type == 'mlp':
                    # Input: [current_row, current_col, maze_id]
                    input_data = torch.tensor([current_pos[0], current_pos[1], maze_id], dtype=torch.float32)
                elif self.model_type == 'cnn':
                    # Input: Maze grid image (1, H, W) with current pos marked
                    input_grid = torch.tensor(maze_grid, dtype=torch.float32).unsqueeze(0) # Add channel dim
                    input_grid[0, current_pos[0], current_pos[1]] = CURRENT_POS_MARKER
                    input_data = input_grid
                elif self.model_type in ['lstm', 'transformer']:
                    # Input: Sequence of coordinates visited so far
                    # Clone to avoid modifying list used by next iteration
                    input_data = torch.stack(current_path_coords).clone()
                else:
                    raise ValueError(f"Unsupported model_type: {self.model_type}")

                all_inputs.append(input_data)
                all_actions.append(action_idx)
                all_positions.append(torch.tensor(current_pos, dtype=torch.float32))
                all_maze_ids.append(maze_id)

        # Collate function will handle padding for sequences
        return {
            "inputs": all_inputs,
            "actions": torch.tensor(all_actions, dtype=torch.long),
            "positions": torch.stack(all_positions),
            "maze_ids": torch.tensor(all_maze_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data["actions"])

    def __getitem__(self, idx):
        return (
            self.data["inputs"][idx],
            self.data["actions"][idx],
            self.data["positions"][idx],
            self.data["maze_ids"][idx]
        )

def pad_collate_fn(batch):
    """Collate function to handle padding for sequence models (LSTM/Transformer)."""
    # Separate inputs, actions, positions, maze_ids
    inputs = [item[0] for item in batch]
    actions = torch.stack([item[1] for item in batch])
    positions = torch.stack([item[2] for item in batch])
    maze_ids = torch.stack([item[3] for item in batch])

    # Check if inputs are sequences (tensors with more than 1 dim)
    if isinstance(inputs[0], torch.Tensor) and inputs[0].ndim > 1 and inputs[0].shape[1] == 2: # Assuming (seq_len, 2) for coords
        # Pad sequences
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0.0) # Pad with (0,0)
    else:
        # Stack non-sequence inputs (MLP, CNN)
        inputs_padded = torch.stack(inputs)

    return inputs_padded, actions, positions, maze_ids

# --- Model Definitions (Skeletons) ---

class MazeMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM, output_dim=NUM_ACTIONS):
        super().__init__()
        # TODO: Implement MLP architecture
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Input x: (batch, 3) -> [r, c, maze_id]
        return self.network(x)

class MazeCNN2D(nn.Module):
    def __init__(self, input_channels=1, height=MAZE_HEIGHT, width=MAZE_WIDTH, hidden_dim=HIDDEN_DIM, output_dim=NUM_ACTIONS):
        super().__init__()
        # TODO: Implement CNN architecture (Conv2d, Pool, Flatten, Linear)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # Calculate flattened size (depends on pooling)
        pooled_h = height // 4
        pooled_w = width // 4
        self.flattened_size = 32 * pooled_h * pooled_w
        self.fc = nn.Linear(self.flattened_size, output_dim)

    def forward(self, x):
        # Input x: (batch, 1, height, width)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        # Add check for flattened size mismatch if needed
        if x.shape[1] != self.flattened_size:
             raise RuntimeError(f"Flattened size mismatch in CNN: Expected {self.flattened_size}, got {x.shape[1]}")
        return self.fc(x)

class MazeLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=HIDDEN_DIM, num_layers=LSTM_LAYERS, output_dim=NUM_ACTIONS):
        super().__init__()
        # TODO: Implement LSTM architecture
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x: (batch, seq_len, 2) -> sequence of (r, c)
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class MazeTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=HIDDEN_DIM, nhead=N_HEAD, num_encoder_layers=TRANSFORMER_LAYERS, max_len=MAX_SEQ_LEN, output_dim=NUM_ACTIONS):
        super().__init__()
        # TODO: Implement Transformer architecture (Input projection, Positional Encoding, Encoder, Output layer)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # Input x: (batch, seq_len, 2)
        seq_len = x.size(1)
        x = self.input_proj(x)
        # Add positional encoding (sliced to current seq_len)
        x = x + self.pos_encoder[:, :seq_len, :]
        memory = self.transformer_encoder(x)
        # Use the output corresponding to the last input token
        last_output = memory[:, -1, :]
        return self.fc(last_output)

# --- Training Loop --- 
def train_model(model, dataloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        for inputs, actions, _, _ in dataloader: # Positions and maze_ids not needed for training
            inputs, actions = inputs.to(device), actions.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) # Get logits
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += actions.size(0)
            epoch_correct += (predicted == actions).sum().item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        epoch_acc = 100 * epoch_correct / epoch_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
             print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    print("Training finished.")
    model.eval() # Set back to eval mode
    return model

# --- Activation Extraction & Visualization --- 

class ActivationExtractor:
    """Manages hook registration and activation extraction for specified layers."""
    def __init__(self, model, layers_to_extract):
        self.model = model
        self.layers_to_extract = layers_to_extract
        self.activations = {} # Stores activations for the current forward pass
        self._hook_handles = []
        self._module_to_name_map = {} # Internal mapping used by hook

    def _hook_fn(self, module, input, output):
        """Hook function to capture activations for the current forward pass."""
        layer_name = self._module_to_name_map.get(module)
        if layer_name:
            activation_data = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name] = activation_data.detach().cpu()

    def _register_hooks(self):
        """Registers forward hooks to specified layers."""
        self.remove_hooks() # Clear previous hooks
        self._module_to_name_map = {}
        for layer_name in self.layers_to_extract:
            try:
                module = self.model.get_submodule(layer_name)
                handle = module.register_forward_hook(self._hook_fn)
                self._hook_handles.append(handle)
                # Store the reverse mapping for the hook to use
                self._module_to_name_map[module] = layer_name
                # print(f"Registered hook for: {layer_name}")
            except AttributeError:
                print(f"Warning: Could not find submodule named '{layer_name}' in {type(self.model).__name__}.")
        # Verify
        found_layers = list(self._module_to_name_map.values())
        not_found = [layer for layer in self.layers_to_extract if layer not in found_layers]
        if not_found:
            print(f"Warning: Failed to register hooks for: {not_found}")

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._module_to_name_map = {}

    def extract(self, dataloader, device):
        """Runs inference and extracts activations across the dataloader."""
        self.model.eval()
        self._register_hooks() # Register hooks before extraction

        collected_activations = {name: [] for name in self.layers_to_extract}
        all_positions_list = []
        all_maze_ids_list = []

        with torch.no_grad():
            analysis_dataloader = DataLoader(dataloader.dataset,
                                             batch_size=dataloader.batch_size,
                                             shuffle=False,
                                             collate_fn=dataloader.collate_fn)

            for inputs, actions, positions, maze_ids in analysis_dataloader:
                self.activations = {} # Reset activations for this batch
                inputs = inputs.to(device)
                _ = self.model(inputs) # Forward pass triggers hooks

                # Collect activations captured by the hook for this batch
                for layer_name in self.layers_to_extract:
                    if layer_name in self.activations:
                        collected_activations[layer_name].append(self.activations[layer_name])
                    # else: # Activation wasn't captured for this layer in this batch
                    #    pass # Or append None/placeholder if needed

                all_positions_list.append(positions.cpu())
                all_maze_ids_list.append(maze_ids.cpu())

        self.remove_hooks() # Clean up hooks afterwards

        # Concatenate collected activations
        final_activations = {}
        min_samples = float('inf')
        for name in self.layers_to_extract:
            if collected_activations[name]:
                try:
                    processed_acts = []
                    for act_tensor in collected_activations[name]:
                        if act_tensor.ndim > 2:
                           # Flatten CNN features, take last token for sequences if needed by vis
                           # Simple flatten for now
                            processed_acts.append(act_tensor.reshape(act_tensor.shape[0], -1))
                        else:
                            processed_acts.append(act_tensor)
                    final_activations[name] = torch.cat(processed_acts, dim=0)
                    min_samples = min(min_samples, final_activations[name].shape[0])
                except Exception as e:
                    print(f"Error concatenating activations for layer '{name}': {e}")
                    final_activations[name] = None
            else:
                # print(f"Warning: No activations collected for layer '{name}'.")
                final_activations[name] = None # Keep as None if nothing was collected

        if not all_positions_list or not all_maze_ids_list:
            return {}, None, None

        all_positions = torch.cat(all_positions_list, dim=0)
        all_maze_ids = torch.cat(all_maze_ids_list, dim=0)

        # Trim if necessary (if some hooks didn't fire consistently)
        valid_activation_count = final_activations[next(iter(final_activations))].shape[0] if any(a is not None for a in final_activations.values()) else 0
        final_sample_count = min(valid_activation_count, all_positions.shape[0])

        if final_sample_count < all_positions.shape[0]:
             print(f"Warning: Trimming data from {all_positions.shape[0]} to {final_sample_count} samples due to inconsistent data collection.")
             all_positions = all_positions[:final_sample_count]
             all_maze_ids = all_maze_ids[:final_sample_count]
             for name in final_activations:
                  if final_activations[name] is not None:
                       final_activations[name] = final_activations[name][:final_sample_count]

        # Filter out layers where no activations were collected at all
        final_activations = {k: v for k, v in final_activations.items() if v is not None}

        return final_activations, all_positions, all_maze_ids

def get_all_activations(model, dataloader, device, layers_to_extract):
     """Wrapper function to use the ActivationExtractor class."""
     extractor = ActivationExtractor(model, layers_to_extract)
     return extractor.extract(dataloader, device)

def plot_dimensionality_reduction(activations, labels, method='tsne', title='Activation Space', color_label='Maze ID', cmap='viridis'):
    """Performs T-SNE or PCA and plots the 2D representation colored by labels."""
    if activations is None or activations.shape[0] == 0:
         print(f"Skipping plot '{title}': No activation data.")
         return
    if activations.shape[0] != labels.shape[0]:
        print(f"Skipping plot '{title}': Mismatch in activation samples ({activations.shape[0]}) and labels ({labels.shape[0]}).")
        return
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1) # Flatten
    if activations.shape[1] <= 1:
        print(f"Skipping plot '{title}': Activation dimension ({activations.shape[1]}) must be > 1.")
        return

    print(f"  Running {method.upper()} on data shape {activations.shape}...")
    activations_np = activations.cpu().numpy()
    labels_np = labels.cpu().numpy()

    if method == 'tsne':
        perplexity = min(max(5, activations_np.shape[0] // 20), 50)
        if activations_np.shape[0] <= perplexity:
            perplexity = max(1, activations_np.shape[0] - 1)
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, n_iter=300, init='pca', learning_rate='auto')
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=SEED)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    try:
        # Scale features before reduction
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        activations_scaled = scaler.fit_transform(activations_np)
        embeddings = reducer.fit_transform(activations_scaled)
    except Exception as e:
        print(f"Error during dimensionality reduction for '{title}': {e}")
        return

    print(f"  Plotting results...")
    plt.figure(figsize=(8, 7))

    # Determine unique labels for colormap normalization
    unique_labels = np.unique(labels_np)
    norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))

    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels_np, cmap=cmap, norm=norm, alpha=0.7, s=10)
    plt.title(title, fontsize=14)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=10)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add colorbar with appropriate ticks
    cbar = plt.colorbar(scatter, ticks=unique_labels if len(unique_labels) < 10 else None)
    cbar.set_label(color_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

# --- Main Execution --- 
def main():
    print(f"Using device: {DEVICE}")

    # 1. Create Datasets and Dataloaders
    print("\n--- Creating Datasets ---")
    datasets = {
        'MLP': MazeNavigationDataset(MAZES, PATHS, model_type='mlp'),
        'CNN': MazeNavigationDataset(MAZES, PATHS, model_type='cnn'),
        'LSTM': MazeNavigationDataset(MAZES, PATHS, model_type='lstm'),
        'Transformer': MazeNavigationDataset(MAZES, PATHS, model_type='transformer')
    }
    # Use pad_collate_fn for all dataloaders to handle different input types uniformly
    dataloaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
                   for name, ds in datasets.items()}
    # Use separate test dataloaders for consistent analysis
    test_dataloaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
                        for name, ds in datasets.items()}
    print("Datasets created.")

    # Print a sample maze
    print("\n--- Sample Maze (U-Shape) ---")
    print(MAZES['U'])

    # 2. Initialize Models
    print("\n--- Initializing Models ---")
    models = {
        'MLP': MazeMLP().to(DEVICE),
        'CNN': MazeCNN2D().to(DEVICE),
        'LSTM': MazeLSTM().to(DEVICE),
        'Transformer': MazeTransformer().to(DEVICE)
    }
    print("Models initialized.")

    # 3. Train Models
    print("\n--- Training Models ---")
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        train_model(model, dataloaders[name], num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE)
        trained_models[name] = model
    print("All models trained.")

    # 4. Activation Analysis
    print("\n--- Activation Analysis --- ")
    # Define layers to extract, focusing on the layer *before* the final output layer
    # We still need to extract these specific layers using their names.
    layers_to_extract = {
        'MLP': ['network.1'],  # Output of ReLU before final Linear
        'CNN': ['pool2'], # Output of last pooling before FC
        'LSTM': ['lstm'], # Output of LSTM layer (last time step handled in extractor)
        'Transformer': ['transformer_encoder.layers.1'] # Output of last Encoder layer (assuming 2 layers)
    }
    # Store the target layer name for easy lookup later
    target_plot_layer = {
        'MLP': 'network.1',
        'CNN': 'pool2',
        'LSTM': 'lstm',
        'Transformer': 'transformer_encoder.layers.1'
    }

    all_model_activations = {}
    all_model_positions = {}
    all_model_maze_ids = {}

    for name, model in trained_models.items():
        print(f"\nExtracting activations for {name}...")
        analysis_loader = test_dataloaders[name]
        layers = layers_to_extract.get(name, [])
        if not layers:
            print(f"  Skipping {name}: No layers defined for extraction.")
            continue

        activations, positions, maze_ids = get_all_activations(model, analysis_loader, DEVICE, layers)

        # Check if activations were actually extracted for this model
        if not activations or all(v is None for v in activations.values()):
             print(f"  Skipping {name}: Failed to extract any activations.")
             continue

        all_model_activations[name] = activations
        all_model_positions[name] = positions
        all_model_maze_ids[name] = maze_ids
        print(f"  Finished extraction for {name}.")

    # Visualize activations for the target layer only
    print("\n--- Visualizing Final Hidden Layer Activations (Per Maze, Colored by Row) ---")
    unique_maze_names = list(MAZES.keys()) # Get maze names for titles

    for model_name, activations_dict in all_model_activations.items():
        target_layer = target_plot_layer.get(model_name)
        if not target_layer or target_layer not in activations_dict or activations_dict[target_layer] is None:
            print(f"Skipping plots for {model_name}: Target layer '{target_layer}' activations not available.")
            continue

        print(f"Processing activations for {model_name} (Layer: {target_layer})...")
        layer_activations = activations_dict[target_layer]
        maze_ids_for_model = all_model_maze_ids.get(model_name)
        positions_for_model = all_model_positions.get(model_name)

        if maze_ids_for_model is None or positions_for_model is None:
            print(f"Skipping plots for {model_name}: Missing maze ID or position data.")
            continue

        # Iterate through each unique maze ID found in the data for this model
        unique_ids = torch.unique(maze_ids_for_model)
        for maze_id in unique_ids:
            maze_id_int = maze_id.item()
            try:
                 maze_name = unique_maze_names[maze_id_int]
            except IndexError:
                 maze_name = f"ID {maze_id_int}"

            print(f"  Generating plot for {model_name} - Maze: {maze_name}...")

            # Filter data for the current maze ID
            mask = (maze_ids_for_model == maze_id)
            filtered_activations = layer_activations[mask]
            filtered_positions = positions_for_model[mask]

            if filtered_activations.shape[0] < 5: # Need enough points for T-SNE
                print(f"    Skipping plot for Maze {maze_name}: Too few data points ({filtered_activations.shape[0]}).")
                continue

            # Plot colored by Row Number
            plot_dimensionality_reduction(
                filtered_activations,
                filtered_positions[:, 0], # Use row index (0) as the label for coloring
                method='tsne',
                title=f'{model_name} - Maze: {maze_name} ({target_layer}) - Colored by Row',
                color_label='Row Number',
                cmap='plasma' # Sequential colormap
            )

            # Optional: Plot colored by Column Number
            # plot_dimensionality_reduction(
            #     filtered_activations,
            #     filtered_positions[:, 1], # Use col index (1) as the label
            #     method='tsne',
            #     title=f'{model_name} - Maze: {maze_name} ({target_layer}) - Colored by Column',
            #     color_label='Column Number',
            #     cmap='viridis'
            # )

    print("\n--- Maze Navigation Script Finished ---")

if __name__ == "__main__":
    # print("Script structure created. Implement TODO sections before execution.")
    main() # Run when ready 