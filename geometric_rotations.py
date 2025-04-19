import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
import random

# --- Configuration ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM_MLP = 4  # x, y, cos(theta), sin(theta)
INPUT_SEQ_LEN_CNN_TRANSFORMER = 4 # Sequence length for CNN/Transformer
OUTPUT_DIM = 2  # x', y'
HIDDEN_DIM = 128
NUM_LAYERS = 2 # Example for MLP/Transformer depth
N_HEAD = 4 # Example for Transformer
KERNEL_SIZE_CNN = 3 # Example for CNN
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_SAMPLES_TRAIN = 10000
NUM_SAMPLES_TEST = 2000

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Dataset ---

class GeometricRotationDataset(Dataset):
    """Generates data for the geometric rotation task."""
    def __init__(self, num_samples, input_repr='mlp'):
        """
        Args:
            num_samples (int): Number of data points to generate.
            input_repr (str): 'mlp', 'cnn', or 'transformer' to format input accordingly.
        """
        self.num_samples = num_samples
        self.input_repr = input_repr
        self.data = self._generate_data()

    def _generate_data(self):
        points_xy = np.random.uniform(-1, 1, size=(self.num_samples, 2)) # Points in [-1, 1] square
        thetas = np.random.uniform(0, 2 * np.pi, size=(self.num_samples, 1))

        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        x = points_xy[:, 0:1]
        y = points_xy[:, 1:2]

        x_prime = x * cos_thetas - y * sin_thetas
        y_prime = x * sin_thetas + y * cos_thetas

        targets = np.hstack((x_prime, y_prime))

        # Prepare inputs based on representation type
        mlp_inputs = np.hstack((x, y, cos_thetas, sin_thetas))

        if self.input_repr == 'mlp':
            inputs = mlp_inputs
        elif self.input_repr in ['cnn', 'transformer']:
            # Reshape for CNN (batch, channels, seq_len) or Transformer (batch, seq_len, features)
            # For CNN (batch, 1, 4) - Channels = 1
            # For Transformer (batch, 4, 1) - Treat each of x,y,cos,sin as a token with feature dim 1
            # Alternatively, Transformer (batch, 4) with embedding layer for dim > 1
            # Let's use (batch, 4, 1) for transformer for now
            inputs = mlp_inputs.reshape(self.num_samples, INPUT_SEQ_LEN_CNN_TRANSFORMER, 1)
            if self.input_repr == 'cnn':
                 # CNN expects (batch, channels, length)
                 inputs = inputs.transpose(0, 2, 1) # -> (batch, 1, 4)

        else:
             raise ValueError("Invalid input_repr specified.")

        return {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
            'angles': torch.tensor(thetas.squeeze(), dtype=torch.float32) # Store original angles for visualization
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data['inputs'][idx], self.data['targets'][idx], self.data['angles'][idx]

# --- Model Definitions ---

class RotationMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM_MLP, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        print(f"MLP initialized with {num_layers} hidden layers.")


    def forward(self, x):
        # Input x shape: (batch_size, input_dim)
        return self.network(x)

class RotationCNN1D(nn.Module):
    def __init__(self, input_channels=1, seq_len=INPUT_SEQ_LEN_CNN_TRANSFORMER, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, kernel_size=KERNEL_SIZE_CNN):
         super().__init__()
         # Example 1D CNN Structure
         self.conv1 = nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=kernel_size, padding=(kernel_size -1)//2)
         self.relu1 = nn.ReLU()
         self.pool1 = nn.MaxPool1d(2) # Reduce sequence length
         self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=kernel_size, padding=(kernel_size -1)//2)
         self.relu2 = nn.ReLU()
         # Calculate flattened size dynamically (careful with pooling/stride)
         # After pool1, length = seq_len // 2 if stride=2
         conv_output_len = seq_len // 2
         self.flattened_size = hidden_dim * conv_output_len
         self.fc = nn.Linear(self.flattened_size, output_dim)
         print(f"CNN1D initialized. Flattened size: {self.flattened_size}")


    def forward(self, x):
        # Input x shape: (batch_size, input_channels, seq_len)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        if x.shape[1] != self.flattened_size:
             # This might happen if seq_len changes or pooling is different
             # Handle adaptively? Or assert? For now, let's try adaptive FC
             # print(f"Warning: Actual flattened size {x.shape[1]} != calculated {self.flattened_size}. Adapting FC.")
             # Recreate FC layer if size mismatch - might be inefficient during training
             # if not hasattr(self, 'fc_adapted') or self.fc_adapted.in_features != x.shape[1]:
             #      self.fc_adapted = nn.Linear(x.shape[1], OUTPUT_DIM).to(x.device)
             # return self.fc_adapted(x)
             # --- OR --- Assert/Error for fixed architecture
             raise RuntimeError(f"Flattened size mismatch in CNN: Expected {self.flattened_size}, got {x.shape[1]}")

        return self.fc(x)


class RotationTransformer(nn.Module):
     def __init__(self, input_feature_dim=1, seq_len=INPUT_SEQ_LEN_CNN_TRANSFORMER, embed_dim=HIDDEN_DIM, nhead=N_HEAD, num_encoder_layers=NUM_LAYERS, output_dim=OUTPUT_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple linear projection instead of Embedding layer for continuous inputs
        self.input_proj = nn.Linear(input_feature_dim, embed_dim)
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.1) # Learnable Positional Encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.output_layer = nn.Linear(embed_dim, output_dim) # Predict from the mean of sequence embeddings

        print(f"Transformer initialized with {num_encoder_layers} encoder layers.")

     def forward(self, x):
        # Input x shape: (batch_size, seq_len, input_feature_dim) e.g., (batch, 4, 1)
        x = self.input_proj(x) # Project features to embed_dim -> (batch, seq_len, embed_dim)
        x = x + self.positional_encoding # Add positional encoding
        x = self.transformer_encoder(x) # -> (batch, seq_len, embed_dim)
        # Aggregate sequence: Mean pooling
        x_pooled = x.mean(dim=1) # -> (batch, embed_dim)
        # --- Alternative: Use first token's output (like BERT's [CLS]) ---
        # x_pooled = x[:, 0] # -> (batch, embed_dim)

        output = self.output_layer(x_pooled) # -> (batch, output_dim)
        return output

# --- Training Loop ---
def train_model(model, dataloader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    model.train() # Set model to training mode
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for inputs, targets, _ in dataloader: # Angles are not needed for training
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0: # Print every 10 epochs
             print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

    print("Training finished.")
    model.eval() # Set model back to evaluation mode
    return model # Return the trained model

# --- Activation Extraction ---

_forward_hooks = []
_activation_dict = {}

def _hook_fn(module, input, output):
    """Hook function to capture activations."""
    global _activation_dict
    # Use a unique identifier for the module (its memory address or a predefined name)
    # For Sequential models, need to map based on predefined names.
    # We will rely on the layer names provided in `layers_to_extract` matching the module structure.
    # Find the name corresponding to this module instance if possible (tricky)
    # For now, let's assume the layer_name is somehow passed or accessible.
    # A simpler approach: Store based on module object directly, map back later.
    # Store output tensor. Detach to prevent holding onto computation graph.
    _activation_dict[module] = output.detach()

def register_hooks(model, layers_to_extract):
    """Registers forward hooks to specified layers."""
    global _forward_hooks
    # Clear previous hooks to avoid duplicates if called multiple times
    for handle in _forward_hooks:
        handle.remove()
    _forward_hooks = []

    registered_modules = {}

    for name, module in model.named_modules():
        if name in layers_to_extract:
            # Check if the layer name exactly matches a module name
            handle = module.register_forward_hook(_hook_fn)
            _forward_hooks.append(handle)
            registered_modules[name] = module
            # print(f"Registered hook for: {name}")
        elif hasattr(model, name) and isinstance(getattr(model, name), nn.Module) and name in layers_to_extract:
             # Handle cases where layer_name is a direct attribute but not in named_modules (less common)
             module_direct = getattr(model, name)
             handle = module_direct.register_forward_hook(_hook_fn)
             _forward_hooks.append(handle)
             registered_modules[name] = module_direct
             # print(f"Registered hook for direct attribute: {name}")

    # Check if all requested layers were found
    found_layers = registered_modules.keys()
    not_found = [layer for layer in layers_to_extract if layer not in found_layers]
    if not_found:
         print(f"Warning: Could not find or register hooks for layers: {not_found}")
         print("Available named modules:", [n for n, m in model.named_modules()])

    return registered_modules # Return mapping from name to module

def get_all_activations(model, dataloader, device, layers_to_extract):
    """Extracts activations from specified layers for all data points.

    Returns:
        dict: {layer_name: activations_tensor}
        torch.Tensor: All target coordinates
        torch.Tensor: All corresponding input angles
    """
    global _activation_dict
    model.eval()
    layer_activations = {name: [] for name in layers_to_extract}
    all_targets_list = []
    all_angles_list = []

    # Register hooks and get the mapping from layer name to module object
    name_to_module_map = register_hooks(model, layers_to_extract)
    # Invert map for easy lookup in the hook
    module_to_name_map = {module: name for name, module in name_to_module_map.items()}

    with torch.no_grad():
        for inputs, targets, angles in dataloader:
            _activation_dict = {} # Clear dict for this batch
            inputs = inputs.to(device)
            _ = model(inputs) # Forward pass triggers hooks

            # Map activations stored by module back to layer names
            batch_activations = {module_to_name_map.get(mod): act.cpu() for mod, act in _activation_dict.items() if mod in module_to_name_map}

            for name in layers_to_extract:
                if name in batch_activations:
                    layer_activations[name].append(batch_activations[name])
                #else: # Handle cases where hook might not have triggered (shouldn't happen if registered)
                    # print(f"Warning: No activation captured for layer '{name}' in this batch.")

            all_targets_list.append(targets.cpu())
            all_angles_list.append(angles.cpu())

    # Remove hooks
    for handle in _forward_hooks:
        handle.remove()

    # Concatenate activations for each layer
    final_activations = {}
    for name in layers_to_extract:
         if layer_activations[name]: # Check if list is not empty
            try:
                final_activations[name] = torch.cat(layer_activations[name], dim=0)
            except Exception as e:
                print(f"Error concatenating activations for layer '{name}': {e}")
                # Optionally print shapes for debugging
                # for i, t in enumerate(layer_activations[name]):
                #     print(f"  Tensor {i} shape: {t.shape}")
                final_activations[name] = None # Mark as problematic
         else:
             print(f"Warning: No activations collected for layer '{name}'.")
             final_activations[name] = None

    all_targets = torch.cat(all_targets_list, dim=0)
    all_angles = torch.cat(all_angles_list, dim=0)

    return final_activations, all_targets, all_angles


# --- Visualization ---
def plot_dimensionality_reduction(activations, labels, method='tsne', title='Activation Space', color_label='Angle (radians)', cmap='hsv'):
    """Performs T-SNE or PCA and plots the 2D representation colored by labels."""
    if activations is None:
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
    if method == 'tsne':
        # Adjust perplexity based on number of samples, minimum 5, max 50
        perplexity = min(max(5, activations.shape[0] // 20), 50)
        if activations.shape[0] <= perplexity:
            perplexity = max(1, activations.shape[0] - 1) # Ensure perplexity < n_samples
            print(f"    Low sample count, adjusting perplexity to {perplexity}")
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, n_iter=300, init='pca')
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=SEED)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    try:
        embeddings = reducer.fit_transform(activations.cpu().numpy()) # Reduce dimensionality
    except Exception as e:
        print(f"Error during dimensionality reduction for '{title}': {e}")
        return

    print(f"  Plotting results...")
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels.cpu().numpy(), cmap=cmap, alpha=0.7, s=10)
    plt.title(title, fontsize=14)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=10)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

def plot_predictions(model, dataset, device=DEVICE, title="Model Predictions"):
    """Plots true vs predicted points for the rotation task."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=min(500, len(dataset)), shuffle=False) # Plot up to 500 points
    inputs, targets, _ = next(iter(dataloader))
    inputs = inputs.to(device)

    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()

    targets = targets.cpu().numpy()

    plt.figure(figsize=(7, 7))
    # Plot true points
    plt.scatter(targets[:, 0], targets[:, 1], label='True Points', alpha=0.6, s=30, marker='o', edgecolor='k')
    # Plot predicted points
    plt.scatter(predictions[:, 0], predictions[:, 1], label='Predicted Points', alpha=0.6, s=30, marker='x', color='red')

    # Optionally draw lines connecting true to predicted (can be messy)
    # for i in range(len(targets)):
    #     plt.plot([targets[i, 0], predictions[i, 0]], [targets[i, 1], predictions[i, 1]], color='gray', linestyle='--', linewidth=0.5)

    plt.title(title, fontsize=14)
    plt.xlabel("X coordinate", fontsize=10)
    plt.ylabel("Y coordinate", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Ensure aspect ratio is equal for rotation visualization
    # Set axis limits slightly larger than [-1, 1] range if points are generated in [-1, 1]
    # The rotated points can go slightly outside [-1, 1]
    limit = 1.5
    plt.xlim([-limit, limit])
    plt.ylim([-limit, limit])
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
def main():
    print(f"Using device: {DEVICE}")

    # 1. Create Datasets and Dataloaders
    print("\\n--- Creating Datasets ---")
    train_datasets = {
        'MLP': GeometricRotationDataset(NUM_SAMPLES_TRAIN, input_repr='mlp'),
        'CNN': GeometricRotationDataset(NUM_SAMPLES_TRAIN, input_repr='cnn'),
        'Transformer': GeometricRotationDataset(NUM_SAMPLES_TRAIN, input_repr='transformer')
    }
    test_datasets = {
        'MLP': GeometricRotationDataset(NUM_SAMPLES_TEST, input_repr='mlp'),
        'CNN': GeometricRotationDataset(NUM_SAMPLES_TEST, input_repr='cnn'),
        'Transformer': GeometricRotationDataset(NUM_SAMPLES_TEST, input_repr='transformer')
    }
    train_dataloaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for name, ds in train_datasets.items()}
    # No shuffling for test loader if we want consistent visualization points
    test_dataloaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for name, ds in test_datasets.items()}
    print("Datasets created.")

    # 2. Initialize Models
    print("\\n--- Initializing Models ---")
    models = {
        'MLP': RotationMLP().to(DEVICE),
        'CNN': RotationCNN1D().to(DEVICE),
        'Transformer': RotationTransformer().to(DEVICE)
    }
    print("Models initialized.")

    # 3. Train Models
    print("\\n--- Training Models ---")
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        trained_models[name] = train_model(model, train_dataloaders[name], num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE)
        print(f"{name} training complete.")
    print("All models trained.")


    # 4. Evaluate and Visualize Predictions
    print("\\n--- Evaluating Predictions ---")
    for name, model in trained_models.items():
        plot_predictions(model, test_datasets[name], device=DEVICE, title=f"{name} Predictions")

    # 5. Extract Activations
    print("\\n--- Extracting Activations ---")
    all_activations = {}
    all_angles = {} # Store angles corresponding to activations

    # Define layers to extract for each model (adjust based on actual model structure)
    layer_map = {
        'MLP': ['network.0', 'network.1', 'network.3', 'network.5'], # Example layers
        'CNN': ['conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'fc'], # Example layers
        'Transformer': ['input_proj', 'transformer_encoder.layers.0.sa_layer_norm', 'transformer_encoder.layers.0.ffn.linear1', 'transformer_encoder.layers.1.sa_layer_norm', 'output_layer'] # Example layers
    }

    for name, model in trained_models.items():
        print(f"Extracting activations for {name}...")
        dataloader = test_dataloaders[name] # Use test set for analysis
        layers_to_extract = layer_map[name]   
        activations, targets, angles = get_all_activations(model, dataloader, DEVICE, layers_to_extract) # Need to modify get_all_activations return
        all_activations[name] = activations
        all_angles[name] = angles
        print(f"Extracted {len(activations)} layers for {name}.")


    # 6. Visualize Activations (T-SNE/PCA colored by Angle)
    print("\\n--- Visualizing Activation Space ---")
    for model_name, activations_dict in all_activations.items():
        print(f"Plotting activations for {model_name}...")
        angles_for_model = all_angles[model_name]
        for layer_name, layer_activations in activations_dict.items():
            if layer_activations is None or layer_activations.ndim < 2 or layer_activations.shape[1] <= 1:
                print(f"  Skipping visualization for layer '{layer_name}': Not suitable.")
                continue
            print(f"  Plotting for layer: {layer_name}")
            plot_dimensionality_reduction(
                layer_activations,
                angles_for_model, # Use angles as labels
                method='tsne', # or 'pca'
                title=f'{model_name} - Layer: {layer_name} - Colored by Angle',
                color_label='Input Angle (radians)',
                cmap='hsv' # Cyclic colormap
            )
    print("Activation visualization skipped (placeholder).")


    print("\\n--- Geometric Rotations Script Finished ---")


if __name__ == "__main__":
    # Placeholder: Fill in the TODOs before running
    print("Script structure created. Implement TODO sections before execution.")
    # main() # Run main when ready 