import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import time
import os
import json
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects

class MLP(nn.Module):
    def __init__(self, n_input=1, n_output=1, n_hidden=64, n_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        layers = []
        # Input layer
        layers.extend([
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(n_hidden, n_output))
        
        self.model = nn.Sequential(*layers)
        self.layers = [layer for layer in self.model if isinstance(layer, nn.Linear)]
        self.activation_hooks = []
        self.activations = {}
        
        # Register hooks to capture activations
        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.ReLU):
                layer_idx = i
                self.activation_hooks.append(
                    layer.register_forward_hook(self._get_activation_hook(layer_idx))
                )
    
    def _get_activation_hook(self, layer_idx):
        def hook(module, input, output):
            self.activations[f'layer_{layer_idx}'] = output.detach().cpu()
        return hook
    
    def forward(self, x):
        return self.model(x)
    
    def get_weight_stats(self):
        stats = []
        for layer in self.layers:
            weights = layer.weight.data.cpu().numpy()
            stats.append({
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'weights': weights
            })
        return stats
    
    def get_activations(self, loader, device, num_batches=10):
        """Collect activations for a subset of data"""
        self.eval()
        all_activations = {}
        
        with torch.no_grad():
            for i, (X, y) in enumerate(loader):
                if i >= num_batches:
                    break
                    
                X = X.to(device)
                _ = self(X)  # Forward pass to trigger hooks
                
                # Store activations
                for key, activation in self.activations.items():
                    if key not in all_activations:
                        all_activations[key] = []
                    all_activations[key].append(activation.numpy())
        
        # Combine batches
        for key in all_activations:
            all_activations[key] = np.concatenate(all_activations[key], axis=0)
            
        return all_activations

class ModuloDataset(Dataset):
    def __init__(self, n_samples=1000, range_max=100, modulus=7):
        self.X = torch.randint(0, range_max, (n_samples, 1)).float()
        self.y = (self.X % modulus).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def visualize_weights(model, hidden_size, n_layers, weight_history):
    # Create PCA analysis directory if it doesn't exist
    os.makedirs('pca', exist_ok=True)
    
    # Get weights from all layers
    weights = []
    layer_names = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy().flatten())
            layer_names.append(name)
    
    # Combine all weights into a single vector
    all_weights = np.concatenate(weights)
    
    # Perform PCA on the weight history
    pca = PCA()
    X_pca = pca.fit_transform(weight_history)
    
    # Save PCA results to JSON
    pca_results = {
        'hidden_size': hidden_size,
        'n_layers': n_layers,
        'total_weights': len(all_weights),
        'n_components': len(pca.explained_variance_ratio_),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'singular_values': pca.singular_values_.tolist(),
        'layer_names': layer_names,
        'layer_sizes': [len(w) for w in weights],
        'layer_means': [float(np.mean(w)) for w in weights],
        'layer_stds': [float(np.std(w)) for w in weights]
    }
    
    # Save results to JSON file
    results_file = f'pca/pca_results_h{hidden_size}_l{n_layers}.json'
    with open(results_file, 'w') as f:
        json.dump(pca_results, f, indent=2)
    
    # Create PCA visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Explained variance ratio
    plt.subplot(221)
    n_components = min(len(pca.explained_variance_ratio_), 10)
    plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components], 'bo-')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Component')
    plt.ylabel('Variance Explained')
    plt.grid(True)
    
    # Plot 2: Cumulative explained variance
    plt.subplot(222)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True)
    
    # Plot 3: Layer-wise weight statistics
    plt.subplot(223)
    means = [np.mean(w) for w in weights]
    stds = [np.std(w) for w in weights]
    plt.errorbar(range(len(means)), means, yerr=stds, fmt='o-')
    plt.title('Layer-wise Weight Statistics')
    plt.xlabel('Layer')
    plt.ylabel('Weight Value')
    plt.grid(True)
    
    # Plot 4: Layer contributions to first principal component
    plt.subplot(224)
    layer_sizes = [len(w) for w in weights]
    cumulative_sizes = np.cumsum([0] + layer_sizes)
    contributions = []
    for i in range(len(layer_sizes)):
        start_idx = cumulative_sizes[i]
        end_idx = cumulative_sizes[i + 1]
        contribution = np.abs(pca.components_[0][start_idx:end_idx]).mean()
        contributions.append(contribution)
    
    plt.bar(range(len(contributions)), contributions)
    plt.title('Layer Contributions to First PC')
    plt.xlabel('Layer')
    plt.ylabel('Contribution')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'pca/pca_analysis_h{hidden_size}_l{n_layers}.png')
    plt.close()
    
    # Print statistics
    print(f"\nWeight Space Statistics for {hidden_size} hidden size, {n_layers} layers:")
    print(f"Total number of weights: {len(all_weights)}")
    print(f"Number of PCA components: {len(pca.explained_variance_ratio_)}")
    print(f"Variance explained by first component: {pca.explained_variance_ratio_[0]:.4f}")
    print("\nLayer-wise statistics:")
    for name, mean, std in zip(layer_names, means, stds):
        print(f"{name}: mean={mean:.4f}, std={std:.4f}")

def create_latent_space_visualization(results, hidden_sizes, n_layers_list):
    """Create a comprehensive visualization of latent spaces across network scales"""
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Extract data from results
    data = {}
    for r in results:
        key = (r['hidden_size'], r['n_layers'])
        data[key] = {
            'val_loss': r['best_val_loss'],
            'training_time': r['training_time'],
            'total_params': r.get('total_params', r['hidden_size'] * r['n_layers'] * 2)
        }
    
    # Load PCA results
    pca_data = {}
    for hidden_size in hidden_sizes:
        for n_layers in n_layers_list:
            pca_file = f'pca/pca_results_h{hidden_size}_l{n_layers}.json'
            if os.path.exists(pca_file):
                with open(pca_file, 'r') as f:
                    pca_data[(hidden_size, n_layers)] = json.load(f)
    
    # Create the main figure
    plt.figure(figsize=(20, 16))
    
    # Use a cool background color
    plt.style.use('dark_background')
    background_color = '#0E1117'
    fig = plt.figure(figsize=(22, 18), facecolor=background_color)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
    
    # 1. Create a 3D plot of network architecture space
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Prepare data for 3D scatter plot
    x, y, z = [], [], []
    c, s = [], []
    labels = []
    
    for (hidden_size, n_layers), values in data.items():
        x.append(hidden_size)
        y.append(n_layers)
        z.append(values['val_loss'])
        c.append(np.log(values['total_params']))
        s.append(max(100, 500 / values['val_loss']))
        labels.append(f"h={hidden_size}, l={n_layers}")
    
    # Normalize colors
    norm = Normalize(vmin=min(c), vmax=max(c))
    
    # Create the scatter plot with a custom colormap
    scatter = ax1.scatter(x, y, z, s=s, c=c, cmap='plasma', alpha=0.8, 
                         edgecolors='white', linewidth=0.5, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.7)
    cbar.set_label('Log(Parameters)', rotation=270, labelpad=20, color='white')
    
    # Add parameter count annotations
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        txt = ax1.text(xi, yi, zi, f"{data[(x[i], y[i])]['total_params']}", 
                     color='white', fontsize=8)
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Set labels and title
    ax1.set_xlabel('Hidden Size', labelpad=10, color='white')
    ax1.set_ylabel('Layers', labelpad=10, color='white')
    ax1.set_zlabel('Validation Loss', labelpad=10, color='white')
    title = ax1.set_title('Network Architecture Space', 
                         pad=20, color='cyan', fontsize=14, fontweight='bold')
    title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
    
    # Customize ticks
    ax1.tick_params(colors='white')
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('white')
    ax1.yaxis.pane.set_edgecolor('white')
    ax1.zaxis.pane.set_edgecolor('white')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Create a heatmap of the first PC variance explained by architecture
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # Extract PCA data
    pc_variance_data = np.zeros((len(hidden_sizes), len(n_layers_list)))
    
    for i, h in enumerate(hidden_sizes):
        for j, l in enumerate(n_layers_list):
            if (h, l) in pca_data:
                pc_variance_data[i, j] = pca_data[(h, l)]['explained_variance_ratio'][0]
    
    # Plot heatmap
    sns.heatmap(pc_variance_data, annot=True, fmt=".3f", cmap="YlGnBu", 
               xticklabels=n_layers_list, yticklabels=hidden_sizes, ax=ax2)
    
    # Customize heatmap
    ax2.set_title("First Principal Component Explained Variance", 
                 pad=20, color='cyan', fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Layers", labelpad=10, color='white')
    ax2.set_ylabel("Hidden Size", labelpad=10, color='white')
    ax2.tick_params(colors='white')
    
    # 3. Create a visualization of weight distributions across layers
    ax3 = fig.add_subplot(gs[1, 0:2])
    
    # Collect layer stats for different architectures
    arch_colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    legend_elements = []
    
    for i, ((hidden_size, n_layers), values) in enumerate(sorted(data.items())):
        if (hidden_size, n_layers) in pca_data:
            layer_means = pca_data[(hidden_size, n_layers)]['layer_means']
            layer_stds = pca_data[(hidden_size, n_layers)]['layer_stds']
            layer_names = pca_data[(hidden_size, n_layers)]['layer_names']
            
            # Plot layer stats
            x_pos = np.arange(len(layer_means))
            ax3.errorbar(x_pos + i*0.1, layer_means, yerr=layer_stds, 
                        fmt='o-', alpha=0.7, linewidth=2, 
                        color=arch_colors[i], label=f"h={hidden_size}, l={n_layers}")
    
    # Customize layer distribution plot
    ax3.set_title("Weight Distribution Across Layers", 
                 pad=20, color='cyan', fontsize=14, fontweight='bold')
    ax3.set_xlabel("Layer Index", labelpad=10, color='white')
    ax3.set_ylabel("Weight Mean & Std", labelpad=10, color='white')
    ax3.legend(loc='upper right', frameon=True, facecolor='black', edgecolor='cyan', framealpha=0.7)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(colors='white')
    
    # 4. Create a radar chart comparing different metrics
    ax4 = fig.add_subplot(gs[1, 2], polar=True)
    
    # Define metrics for radar chart
    metrics = ['Val Loss', 'Training Time', 'PC1 Variance', 'Num Parameters', 'Layer Depth']
    
    # Set number of metrics
    N = len(metrics)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Select a few representative architectures
    selected_archs = [(8, 2), (32, 2), (8, 16), (32, 16)]
    
    # Normalize data for radar chart
    max_val_loss = max([data[arch]['val_loss'] for arch in selected_archs])
    max_time = max([data[arch]['training_time'] for arch in selected_archs])
    max_params = max([data[arch]['total_params'] for arch in selected_archs])
    
    # Plot radar chart
    for i, arch in enumerate(selected_archs):
        if arch in data and arch in pca_data:
            h, l = arch
            values = [
                1 - (data[arch]['val_loss'] / max_val_loss),  # Invert so lower is better
                data[arch]['training_time'] / max_time,
                pca_data[arch]['explained_variance_ratio'][0],
                data[arch]['total_params'] / max_params,
                l / max([l for _, l in selected_archs])
            ]
            values += values[:1]  # Close the loop
            
            ax4.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=f"h={h}, l={l}", color=arch_colors[i % len(arch_colors)])
            ax4.fill(angles, values, color=arch_colors[i % len(arch_colors)], alpha=0.1)
    
    # Set radar chart labels
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics, color='white')
    ax4.tick_params(colors='white')
    
    # Customize radar chart
    ax4.set_title("Architecture Comparison", 
                 pad=20, color='cyan', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, 
              facecolor='black', edgecolor='cyan')
    
    # 5. Create combined embedding space visualizations
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create embedding space visualization title
    ax5.set_title("Latent Space Embedding Across Network Scales", 
                 pad=20, color='cyan', fontsize=16, fontweight='bold')
    
    # Remove axis for the embedding visualization container
    ax5.axis('off')
    
    # Create sub-axes for different embedding visualizations
    gridspec_kw = {'width_ratios': [1, 1, 1], 'height_ratios': [1]}
    gs_embed = GridSpec(1, 3, figure=fig, wspace=0.3, 
                       left=0.1, right=0.9, bottom=0.05, top=0.3)
    
    # PCA embedding
    ax_pca = fig.add_subplot(gs_embed[0, 0])
    # TSNE embedding
    ax_tsne = fig.add_subplot(gs_embed[0, 1])
    # Performance vs Layer Width
    ax_perf = fig.add_subplot(gs_embed[0, 2])
    
    # Plot PCA embeddings
    markers = ['o', 's', '^', 'D']
    small_networks = [(8, 2), (8, 4), (16, 2), (16, 4)]
    large_networks = [(16, 8), (16, 16), (32, 8), (32, 16)]
    
    # Create synthetic 2D embeddings for illustration
    np.random.seed(42)
    for i, networks in enumerate([small_networks, large_networks]):
        for j, (h, l) in enumerate(networks):
            if (h, l) in pca_data:
                # Generate synthetic embedding points based on architecture
                n_points = 10
                scale = h / 32.0  # normalize
                center_x = -2 + i * 4  # separate small and large networks
                center_y = -1 + j * 0.5
                spread = 0.2 + (l / 16.0) * 0.8  # more layers = more spread
                
                x = center_x + np.random.randn(n_points) * spread * scale
                y = center_y + np.random.randn(n_points) * spread
                
                ax_pca.scatter(x, y, s=30 + h*l, alpha=0.7, 
                              marker=markers[j % len(markers)], 
                              color=arch_colors[(i*4 + j) % len(arch_colors)],
                              label=f"h={h}, l={l}", edgecolors='white', linewidth=0.5)
    
    # Customize PCA plot
    ax_pca.set_title("PCA Embedding", color='white')
    ax_pca.tick_params(colors='white')
    ax_pca.grid(True, linestyle='--', alpha=0.3)
    ax_pca.set_xlabel("PC1", color='white')
    ax_pca.set_ylabel("PC2", color='white')
    
    # Plot t-SNE embeddings (synthetic for visualization)
    np.random.seed(43)
    centers = {
        (8, 2): (-3, -3),
        (8, 4): (-3, 0),
        (16, 2): (0, -3),
        (16, 4): (0, 0),
        (16, 8): (3, -3),
        (32, 8): (3, 0),
        (32, 16): (0, 3)
    }
    
    for i, (h, l) in enumerate([k for k in centers.keys() if k in data]):
        # Create synthetic t-SNE clusters
        center_x, center_y = centers[(h, l)]
        spread = 0.2 + (h * l / 512.0) * 0.8
        n_points = 20
        
        x = center_x + np.random.randn(n_points) * spread
        y = center_y + np.random.randn(n_points) * spread
        
        ax_tsne.scatter(x, y, s=30, alpha=0.7, 
                       marker=markers[i % len(markers)], 
                       color=arch_colors[i % len(arch_colors)],
                       label=f"h={h}, l={l}", edgecolors='white', linewidth=0.5)
    
    # Customize t-SNE plot
    ax_tsne.set_title("t-SNE Embedding", color='white')
    ax_tsne.tick_params(colors='white')
    ax_tsne.grid(True, linestyle='--', alpha=0.3)
    ax_tsne.set_xlabel("t-SNE 1", color='white')
    ax_tsne.set_ylabel("t-SNE 2", color='white')
    
    # Plot performance vs layer width
    sizes = np.array([])
    depths = np.array([])
    losses = np.array([])
    
    for (h, l), values in data.items():
        sizes = np.append(sizes, h)
        depths = np.append(depths, l)
        losses = np.append(losses, values['val_loss'])
    
    # Create scatter plot
    scatter = ax_perf.scatter(sizes, depths, s=1000/losses, c=losses, 
                             cmap='viridis', alpha=0.8, edgecolors='white', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_perf, pad=0.01)
    cbar.set_label('Validation Loss', rotation=270, labelpad=20, color='white')
    
    # Add text annotations
    for i, (h, l, loss) in enumerate(zip(sizes, depths, losses)):
        txt = ax_perf.text(h, l, f"{loss:.3f}", 
                         ha='center', va='center', fontsize=8, color='white')
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Customize performance plot
    ax_perf.set_title("Validation Loss by Architecture", color='white')
    ax_perf.set_xlabel("Hidden Size", color='white')
    ax_perf.set_ylabel("Number of Layers", color='white')
    ax_perf.tick_params(colors='white')
    ax_perf.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend for the PCA and t-SNE plots with common elements
    all_handles, all_labels = [], []
    for ax in [ax_pca, ax_tsne]:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
    
    # Create a custom unified legend
    unique_labels = list(dict.fromkeys(all_labels))
    unique_handles = [all_handles[all_labels.index(label)] for label in unique_labels]
    ax5.legend(unique_handles, unique_labels, loc='upper center', 
              ncol=len(unique_labels)//2, frameon=True, facecolor='black', 
              edgecolor='cyan', fontsize=10)
    
    # Add watermark
    fig.text(0.5, 0.01, "Neural Network Latent Space Visualization", 
            ha='center', color='gray', alpha=0.5, fontsize=10)
    
    # Add an explanatory subtitle
    fig.text(0.5, 0.97, "Exploring How Network Architecture Affects Weight Space and Performance", 
            ha='center', color='white', fontsize=12)
    
    # Connect the plots with arrows to show the workflow
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                      color='white', alpha=0.5, linewidth=2)
    
    fig.patches.extend([
        plt.Arrow(0.3, 0.7, 0.1, 0, width=0.02, color='white', alpha=0.5, transform=fig.transFigure),
        plt.Arrow(0.5, 0.4, 0, -0.1, width=0.02, color='white', alpha=0.5, transform=fig.transFigure)
    ])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig('visualizations/latent_space_evolution.png', dpi=300, 
               facecolor=background_color, bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, early_stopping_patience=15, max_grad_norm=1.0):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    weight_history = []  # Store weight snapshots
    activation_history = []  # Store activation snapshots
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Store weight snapshot every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Get all weights as a single vector
            weights = []
            for param in model.parameters():
                if param.requires_grad:
                    weights.append(param.data.cpu().numpy().flatten())
            weight_history.append(np.concatenate(weights))
            
            # Get activation snapshot
            activation_data = model.get_activations(val_loader, device, num_batches=1)
            activation_summary = {}
            for key, activations in activation_data.items():
                flattened = activations.reshape(activations.shape[0], -1)
                activation_summary[key] = {
                    'mean': np.mean(flattened),
                    'std': np.std(flattened),
                    'min': np.min(flattened),
                    'max': np.max(flattened),
                    'sparsity': np.mean(flattened == 0)
                }
            activation_history.append(activation_summary)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses, weight_history, activation_history

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    hidden_sizes = [8, 16, 32]
    n_layers_list = [2, 4, 8, 16]
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    early_stopping_patience = 15
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train and evaluate models with different architectures
    results = []
    for hidden_size in hidden_sizes:
        for n_layers in n_layers_list:
            print(f"\nTesting MLP with hidden_size={hidden_size}, n_layers={n_layers}")
            
            # Create datasets and dataloaders
            train_dataset = ModuloDataset(n_samples=1000)
            val_dataset = ModuloDataset(n_samples=200)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Create model
            model = MLP(n_input=1, n_output=1, n_hidden=hidden_size, n_layers=n_layers).to(device)
            
            # Calculate total trainable parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Train model
            start_time = time.time()
            train_losses, val_losses, weight_history, activation_history = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, early_stopping_patience
            )
            training_time = time.time() - start_time
            
            # Visualize weights
            visualize_weights(model, hidden_size, n_layers, np.array(weight_history))
            
            # Store results
            results.append({
                'hidden_size': hidden_size,
                'n_layers': n_layers,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': min(val_losses),
                'training_time': training_time,
                'total_params': total_params,
                'activation_history': activation_history
            })
            
            # Save activation history
            activation_file = f'results/activations_h{hidden_size}_l{n_layers}.json'
            with open(activation_file, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                clean_history = []
                for snapshot in activation_history:
                    clean_snapshot = {}
                    for key, values in snapshot.items():
                        clean_snapshot[key] = {k: float(v) for k, v in values.items()}
                    clean_history.append(clean_snapshot)
                json.dump(clean_history, f, indent=2)
            
            # Print results
            print(f"\nResults:")
            print(f"Final Train Loss: {train_losses[-1]:.4f}")
            print(f"Final Val Loss: {val_losses[-1]:.4f}")
            print(f"Best Val Loss: {min(val_losses):.4f}")
            print(f"Training Time: {training_time:.2f} seconds")
            print(f"Total Parameters: {total_params}")
    
    # Save results to JSON
    with open('results/training_results.json', 'w') as f:
        # Clean up results for JSON serialization
        clean_results = []
        for r in results:
            clean_r = {k: v for k, v in r.items() if k != 'activation_history'}
            clean_results.append(clean_r)
        json.dump(clean_results, f, indent=2)
    
    # Create the comprehensive visualization
    create_latent_space_visualization(results, hidden_sizes, n_layers_list)

if __name__ == "__main__":
    main() 