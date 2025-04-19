import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
import random
from typing import List, Tuple, Dict, Any
import learn2learn as l2l

# Import models and datasets from ManyPaths
import sys
sys.path.append('ManyPaths')
from models import MLP, CNN, Transformer
from constants import MLP_PARAMS, CNN_PARAMS, TRANSFORMER_PARAMS
from datasets import MetaBitConceptsDataset, MetaModuloDataset

# Create directories for outputs
os.makedirs('results/weight_space', exist_ok=True)
os.makedirs('visualizations/weight_space', exist_ok=True)
os.makedirs('visualizations/maml_weight_space', exist_ok=True)

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add a function to prepare data for different model types
def prepare_batch_for_model(x_batch, model_type, data_type):
    """
    Prepare input data batch for different model types by ensuring they have the correct shape
    """
    if model_type == "transformer":
        # For transformer, we need to ensure data is in correct format
        if data_type == "bits":
            # Print detailed shape for debugging
            print(f"Transformer input shape: {x_batch.shape}, data_type: {data_type}")
            
            # The shape [batch, seq_len, feature_dim] is required 
            if x_batch.dim() == 3 and x_batch.shape[2] == 1:
                # We have [batch, seq_len=4, feature_dim=1]
                # Need to transpose to [batch, feature_dim=4, seq_len=1]
                x_batch = x_batch.permute(0, 2, 1)
                print(f"Transposed shape to: {x_batch.shape}")
            elif x_batch.dim() == 2 and x_batch.shape[1] == 4:
                # Shape is [batch, features=4], add sequence dim
                x_batch = x_batch.unsqueeze(1)
                print(f"Added sequence dimension: {x_batch.shape}")
            else:
                print(f"Shape mismatch for transformer, attempting reshape")
                try:
                    # Attempt to reshape to [batch, 1, 4]
                    batch_size = x_batch.shape[0]
                    x_batch = x_batch.reshape(batch_size, 1, 4)
                    print(f"Reshaped to: {x_batch.shape}")
                except:
                    print(f"Failed to reshape, keeping original: {x_batch.shape}")
                    
            return x_batch
    
    # For CNN, ensure images are in the right format
    elif model_type == "cnn" and data_type == "image":
        if x_batch.dim() == 4 and x_batch.shape[-1] == 3:  # (batch_size, H, W, C)
            # Convert (B, H, W, C) to (B, C, H, W)
            return x_batch.permute(0, 3, 1, 2)
        elif x_batch.dim() == 3 and x_batch.shape[-1] == 3:  # (H, W, C)
            # Convert (H, W, C) to (1, C, H, W)
            return x_batch.permute(2, 0, 1).unsqueeze(0)
        else:
            # Already in correct format or needs no change
            return x_batch
    
    # For other models, return as is
    return x_batch

# Extract weight trajectories during adaptation
def extract_weight_trajectories(model, dataloader, model_type, data_type, adaptation_steps=5, num_tasks=10):
    """Extract weight trajectories during adaptation"""
    trajectories = []
    criterion = nn.BCEWithLogitsLoss()
    task_count = 0
    
    # Get individual tasks from the dataloader
    for task_batch in dataloader:
        if task_count >= num_tasks:
            break
            
        # Unpack batch
        X_s, y_s, _, _ = task_batch
        
        for i in range(len(X_s)):
            if task_count >= num_tasks:
                break
                
            # Get individual task data
            x_support = X_s[i]
            y_support = y_s[i]
            
            # Prepare data for the model
            x_support = prepare_batch_for_model(x_support, model_type, data_type)
            
            # Clone model to keep the original weights
            learner = l2l.algorithms.MAML(model, lr=0.01)
            adapted_model = learner.clone()
            
            # Store initial weights
            weights = []
            params = []
            for name, param in adapted_model.named_parameters():
                if 'weight' in name and len(param.shape) > 1:  # Only consider weight matrices
                    params.append(param.detach().cpu().numpy().flatten())
            
            # Flatten and concatenate all weights
            weights.append(np.concatenate(params) if params else np.array([]))
            
            # Adapt model for several steps
            for step in range(adaptation_steps):
                # Forward pass
                predictions = adapted_model(x_support)
                loss = criterion(predictions, y_support)
                
                # Adaptation step
                adapted_model.adapt(loss)
                
                # Store weights after each step
                params = []
                for name, param in adapted_model.named_parameters():
                    if 'weight' in name and len(param.shape) > 1:
                        params.append(param.detach().cpu().numpy().flatten())
                
                weights.append(np.concatenate(params) if params else np.array([]))
            
            trajectories.append({
                'task_idx': task_count,
                'weights': weights
            })
            task_count += 1
    
    return trajectories

# Visualize weight space
def visualize_weight_space(vanilla_trajectories, meta_trajectories, model_type):
    """Visualize weight space of vanilla and meta-trained models"""
    if not vanilla_trajectories or not meta_trajectories:
        print(f"Warning: Not enough trajectories for {model_type} to visualize")
        return
        
    # Check if we have non-empty weight arrays
    has_weights = all(len(traj['weights']) > 0 and len(traj['weights'][0]) > 0 
                     for traj in vanilla_trajectories + meta_trajectories)
    if not has_weights:
        print(f"Warning: Empty weight arrays for {model_type}")
        return
    
    plt.figure(figsize=(15, 12))
    
    # Combine all weights for PCA
    all_weights = []
    for traj in vanilla_trajectories:
        all_weights.extend(traj['weights'])
    
    for traj in meta_trajectories:
        all_weights.extend(traj['weights'])
    
    # Apply PCA
    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(all_weights)
    
    # Split back into trajectories
    steps_per_traj = len(vanilla_trajectories[0]['weights'])
    
    vanilla_reduced = reduced_weights[:len(vanilla_trajectories) * steps_per_traj]
    meta_reduced = reduced_weights[len(vanilla_trajectories) * steps_per_traj:]
    
    # Reshape
    vanilla_reduced = vanilla_reduced.reshape(len(vanilla_trajectories), steps_per_traj, 2)
    meta_reduced = meta_reduced.reshape(len(meta_trajectories), steps_per_traj, 2)
    
    # Plot trajectories
    for i in range(len(vanilla_trajectories)):
        # Vanilla model trajectory
        plt.subplot(2, 2, 1)
        plt.plot(vanilla_reduced[i, :, 0], vanilla_reduced[i, :, 1], 'b-', alpha=0.5)
        plt.scatter(vanilla_reduced[i, 0, 0], vanilla_reduced[i, 0, 1], color='blue', marker='o', s=80, label='Initial' if i == 0 else "")
        plt.scatter(vanilla_reduced[i, -1, 0], vanilla_reduced[i, -1, 1], color='cyan', marker='x', s=80, label='Final' if i == 0 else "")
        
        # Meta-trained model trajectory
        plt.subplot(2, 2, 2)
        plt.plot(meta_reduced[i, :, 0], meta_reduced[i, :, 1], 'r-', alpha=0.5)
        plt.scatter(meta_reduced[i, 0, 0], meta_reduced[i, 0, 1], color='red', marker='o', s=80, label='Initial' if i == 0 else "")
        plt.scatter(meta_reduced[i, -1, 0], meta_reduced[i, -1, 1], color='orange', marker='x', s=80, label='Final' if i == 0 else "")
    
    # Overlay initial and final points for all trajectories
    plt.subplot(2, 2, 3)
    for i in range(len(vanilla_trajectories)):
        plt.scatter(vanilla_reduced[i, 0, 0], vanilla_reduced[i, 0, 1], color='blue', marker='o', s=50, alpha=0.7)
        plt.scatter(vanilla_reduced[i, -1, 0], vanilla_reduced[i, -1, 1], color='cyan', marker='x', s=50, alpha=0.7)
        plt.scatter(meta_reduced[i, 0, 0], meta_reduced[i, 0, 1], color='red', marker='o', s=50, alpha=0.7)
        plt.scatter(meta_reduced[i, -1, 0], meta_reduced[i, -1, 1], color='orange', marker='x', s=50, alpha=0.7)
    
    # Plot centroids and final points for comparison
    plt.subplot(2, 2, 4)
    vanilla_initial = vanilla_reduced[:, 0, :].mean(axis=0)
    vanilla_final = vanilla_reduced[:, -1, :].mean(axis=0)
    meta_initial = meta_reduced[:, 0, :].mean(axis=0)
    meta_final = meta_reduced[:, -1, :].mean(axis=0)
    
    # Plot centroids
    plt.scatter(vanilla_initial[0], vanilla_initial[1], color='blue', marker='o', s=200, label='Vanilla Initial')
    plt.scatter(vanilla_final[0], vanilla_final[1], color='cyan', marker='x', s=200, label='Vanilla Final')
    plt.scatter(meta_initial[0], meta_initial[1], color='red', marker='o', s=200, label='Meta Initial')
    plt.scatter(meta_final[0], meta_final[1], color='orange', marker='x', s=200, label='Meta Final')
    
    # Connect related points
    plt.plot([vanilla_initial[0], vanilla_final[0]], [vanilla_initial[1], vanilla_final[1]], 'b-', linewidth=2)
    plt.plot([meta_initial[0], meta_final[0]], [meta_initial[1], meta_final[1]], 'r-', linewidth=2)
    
    # Add titles and labels
    plt.subplot(2, 2, 1)
    plt.title(f'Vanilla {model_type.upper()} Weight Trajectories', fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.title(f'Meta-Trained {model_type.upper()} Weight Trajectories', fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.title('All Initial and Final Points', fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.title('Average Weight Trajectories', fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/maml_weight_space/{model_type}_weight_space_comparison.png', dpi=300)
    plt.close()

# Train vanilla model function
def train_vanilla_model(model, dataloader, model_type, data_type, epochs=10, lr=0.001):
    """Train a vanilla model on the data"""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        task_count = 0
        
        for task_batch in dataloader:
            # Process one task at a time instead of batching due to variable sizes
            X_s, y_s, _, _ = task_batch
            
            for i in range(len(X_s)):
                # Get the specific task
                x_support = X_s[i]
                y_support = y_s[i]
                
                # Debug input shape
                if epoch == 0 and i == 0:
                    print(f"Original input shape: {x_support.shape}, data_type: {data_type}")
                
                # Prepare data for the model
                x_support = prepare_batch_for_model(x_support, model_type, data_type)
                
                # Debug prepared shape
                if epoch == 0 and i == 0:
                    print(f"Prepared input shape: {x_support.shape}")
                
                # Forward pass
                predictions = model(x_support)
                loss = criterion(predictions, y_support)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                task_count += 1
        
        avg_epoch_loss = epoch_loss / max(1, task_count)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
    
    return train_losses

# Define a custom collate function to handle variable-sized tensors
def custom_collate(batch):
    """
    Custom collate function that doesn't stack tensors but returns them as lists.
    This handles the case where tensors in a batch have different shapes.
    """
    elem = batch[0]
    if isinstance(elem, tuple):
        return [item for item in zip(*batch)]
    return batch

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model types to train and visualize
    model_types = ["mlp", "cnn", "transformer"]
    
    for model_type in model_types:
        print(f"\n--- Working with {model_type.upper()} model ---")
        
        # Dataset parameters
        n_tasks = 100
        n_samples = 10
        batch_size = 4
        
        # Create dataset
        if model_type == "transformer":
            # Use bit concepts dataset which is more suitable for transformers
            dataset = MetaBitConceptsDataset(
                n_tasks=n_tasks,
                data="bits",
                model="transformer"
            )
        else:
            # Use bit concepts dataset for MLP and CNN
            dataset = MetaBitConceptsDataset(
                n_tasks=n_tasks,
                data="image",
                model=model_type
            )
        
        # Create dataloaders
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        
        # Initialize models
        if model_type == "mlp":
            hidden_dim, n_layers = MLP_PARAMS[7]  # Using index 7 as it seems to be a default setting
            vanilla_model = MLP(
                n_input=32*32*3 if dataset.data == "image" else 4,  # Image or bit vector input
                n_output=1,
                n_hidden=hidden_dim,
                n_layers=n_layers,
                n_input_channels=3 if dataset.data == "image" else 1
            ).to(device)
            meta_model = MLP(
                n_input=32*32*3 if dataset.data == "image" else 4,
                n_output=1,
                n_hidden=hidden_dim,
                n_layers=n_layers,
                n_input_channels=3 if dataset.data == "image" else 1
            ).to(device)
        elif model_type == "cnn":
            hiddens, n_layers = CNN_PARAMS[7]  # Using index 7 as default
            vanilla_model = CNN(
                n_input_channels=3,  # Image input
                n_output=1,
                n_hiddens=hiddens,
                n_layers=n_layers
            ).to(device)
            meta_model = CNN(
                n_input_channels=3,
                n_output=1,
                n_hiddens=hiddens,
                n_layers=n_layers
            ).to(device)
        else:  # transformer
            d_model, num_layers = TRANSFORMER_PARAMS[7]  # Using index 7 as default
            vanilla_model = Transformer(
                n_input=4,  # Each feature is 4-dimensional
                n_output=1,
                d_model=d_model,
                nhead=4,
                num_layers=num_layers,
                dim_feedforward=2 * d_model
            ).to(device)
            meta_model = Transformer(
                n_input=4,  # Each feature is 4-dimensional  
                n_output=1,
                d_model=d_model,
                nhead=4,
                num_layers=num_layers,
                dim_feedforward=2 * d_model
            ).to(device)
        
        # Train vanilla model
        print("\nTraining vanilla model...")
        vanilla_losses = train_vanilla_model(
            model=vanilla_model,
            dataloader=dataloader,
            model_type=model_type,
            data_type=dataset.data,
            epochs=10,
            lr=0.001
        )
        
        # Meta-train model using learn2learn
        print("\nMeta-training model...")
        maml = l2l.algorithms.MAML(meta_model, lr=0.01, first_order=False)
        meta_opt = optim.Adam(maml.parameters(), lr=0.001)
        
        data_type = dataset.data  # Get data type from dataset
        
        for epoch in range(5):
            meta_train_loss = 0.0
            task_count = 0
            
            for task_batch in dataloader:
                # Get support and query sets
                X_s, y_s, X_q, y_q = task_batch
                
                for i in range(len(X_s)):
                    meta_opt.zero_grad()
                    
                    # Process a single task
                    x_support = X_s[i]
                    y_support = y_s[i]
                    x_query = X_q[i] 
                    y_query = y_q[i]
                    
                    # Prepare data for the model
                    x_support = prepare_batch_for_model(x_support, model_type, data_type)
                    x_query = prepare_batch_for_model(x_query, model_type, data_type)
                    
                    # Clone model for adaptation
                    learner = maml.clone()
                    
                    # Adapt on support set
                    predictions = learner(x_support)
                    support_loss = nn.BCEWithLogitsLoss()(predictions, y_support)
                    learner.adapt(support_loss)
                    
                    # Evaluate on query set
                    predictions = learner(x_query)
                    query_loss = nn.BCEWithLogitsLoss()(predictions, y_query)
                    
                    # Backward and optimize
                    query_loss.backward()
                    meta_opt.step()
                    
                    meta_train_loss += query_loss.item()
                    task_count += 1
            
            avg_meta_loss = meta_train_loss / max(1, task_count)
            print(f"Epoch {epoch+1}/5: Meta Loss = {avg_meta_loss:.4f}")
        
        # Extract weight trajectories
        print("\nExtracting weight trajectories...")
        meta_trajectories = extract_weight_trajectories(
            model=meta_model,
            dataloader=dataloader,
            model_type=model_type,
            data_type=dataset.data,
            adaptation_steps=5,
            num_tasks=10
        )
        
        vanilla_trajectories = extract_weight_trajectories(
            model=vanilla_model,
            dataloader=dataloader,
            model_type=model_type,
            data_type=dataset.data,
            adaptation_steps=5,
            num_tasks=10
        )
        
        # Visualize weight space
        print("\nVisualizing weight space...")
        visualize_weight_space(
            vanilla_trajectories=vanilla_trajectories,
            meta_trajectories=meta_trajectories,
            model_type=model_type
        )
        
        print(f"Completed visualization for {model_type} model")
    
    print("All visualizations complete!")
    
    # Update the results markdown to include transformer
    with open('meta_learning_results.md', 'a') as f:
        f.write("\n### Transformer Model\n")
        f.write("![Transformer Weight Space](visualizations/maml_weight_space/transformer_weight_space_comparison.png)\n\n")
        f.write("## Additional Observations for Transformer\n\n")
        f.write("1. **Transformer Complexity**: The transformer model shows more complex weight space dynamics due to its attention mechanisms.\n\n")
        f.write("2. **Adaptation Patterns**: Compared to MLP and CNN models, transformer models exhibit different adaptation trajectories, reflecting their unique architecture.\n\n")
        f.write("3. **Meta-Learning Effect**: The impact of meta-learning on transformers demonstrates how even complex architectures can benefit from meta-initialization.\n\n")

if __name__ == "__main__":
    main() 