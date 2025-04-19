import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import glob

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Define model colors and markers
MODEL_COLORS = {
    "mlp": "#4C72B0",   # Blue
    "cnn": "#C44E52",   # Red
}

# Size markers based on hidden size or number of layers
def get_size_marker(model_type, size):
    if model_type == 'mlp':
        if size <= 32:
            return 'o'  # Small
        elif size <= 64:
            return 's'  # Medium
        elif size <= 128:
            return '^'  # Large
        else:
            return 'D'  # XLarge
    else:  # CNN
        if size == 1:
            return 'o'  # Small
        elif size == 2:
            return 's'  # Medium
        elif size == 3:
            return '^'  # Large
        else:
            return 'D'  # XLarge

def load_all_activations():
    """Load all activation files from results/activations directory"""
    all_results = {}
    
    # Get all .npz files
    activation_files = glob.glob('results/activations/*.npz')
    
    for file_path in activation_files:
        # Extract model type and size from filename
        filename = os.path.basename(file_path)
        model_type, size_str = filename.split('_')
        size = int(size_str.split('.')[0])
        
        # Load the data
        data = np.load(file_path)
        activations = data['activations']
        true_labels = data['true_labels']
        predicted_labels = data['predicted_labels']
        
        # Store in dictionary
        if model_type not in all_results:
            all_results[model_type] = {}
        
        all_results[model_type][size] = {
            'activations': activations,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }
    
    return all_results

def calculate_concept_separation(activations, labels):
    """Calculate separation between concept classes in the activation space"""
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Calculate centroids for each class
    centroids = {}
    for label in unique_labels:
        centroids[label] = np.mean(activations[labels == label], axis=0)
    
    # Calculate average intra-class distance
    intra_class_distances = []
    for label in unique_labels:
        # Get points for this class
        class_points = activations[labels == label]
        
        # Skip if only one point
        if len(class_points) <= 1:
            continue
            
        # Calculate distances from each point to class centroid
        centroid = centroids[label]
        distances = np.linalg.norm(class_points - centroid, axis=1)
        intra_class_distances.extend(distances)
    
    # Calculate average inter-class distance
    inter_class_distances = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            distance = np.linalg.norm(centroids[label1] - centroids[label2])
            inter_class_distances.append(distance)
    
    # Calculate metrics
    avg_intra_class = np.mean(intra_class_distances) if intra_class_distances else 0
    avg_inter_class = np.mean(inter_class_distances) if inter_class_distances else 0
    
    # Avoid division by zero
    if avg_intra_class > 0:
        separation_ratio = avg_inter_class / avg_intra_class
    else:
        separation_ratio = float('inf')
    
    return {
        'intra_class': avg_intra_class,
        'inter_class': avg_inter_class,
        'separation_ratio': separation_ratio
    }

def create_latent_space_visualization(all_results):
    """Create a comprehensive visualization of latent space representations"""
    # Create a large figure for all visualizations
    plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = GridSpec(2, 3, height_ratios=[1, 1.5])
    
    # Plot 1: Parameter count vs. concept separation
    ax1 = plt.subplot(gs[0, 0:2])
    
    # For each model type (MLP, CNN)
    for model_type in all_results.keys():
        model_sizes = []
        separation_ratios = []
        marker_styles = []
        
        for size, data in all_results[model_type].items():
            activations = data['activations']
            true_labels = data['true_labels']
            
            # Calculate concept separation
            separation = calculate_concept_separation(activations, true_labels)
            
            # Get marker style based on size
            marker = get_size_marker(model_type, size)
            
            # Store data for plotting
            model_sizes.append(size)
            separation_ratios.append(separation['separation_ratio'])
            marker_styles.append(marker)
        
        # Sort by size for line plot
        sorted_indices = np.argsort(model_sizes)
        sorted_sizes = [model_sizes[i] for i in sorted_indices]
        sorted_ratios = [separation_ratios[i] for i in sorted_indices]
        sorted_markers = [marker_styles[i] for i in sorted_indices]
        
        # Plot line
        ax1.plot(sorted_sizes, sorted_ratios, 
                color=MODEL_COLORS[model_type], 
                label=f"{model_type.upper()}",
                linestyle='-', linewidth=2)
        
        # Plot points with different markers
        for size, ratio, marker in zip(sorted_sizes, sorted_ratios, sorted_markers):
            ax1.scatter(size, ratio, 
                      color=MODEL_COLORS[model_type], 
                      marker=marker, s=150, 
                      edgecolor='white', linewidth=1)
    
    # Format plot
    ax1.set_title("Concept Separation vs. Model Size", fontsize=14, color='black')
    ax1.set_xlabel("Model Size (Hidden Units for MLP, Layers for CNN)", fontsize=12, color='black')
    ax1.set_ylabel("Separation Ratio (Higher is Better)", fontsize=12, color='black')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Plot 2: Silhouette scores comparison
    ax3 = plt.subplot(gs[0, 2])
    
    # For each model type
    for model_type in all_results.keys():
        model_sizes = []
        silhouette_scores = []
        
        for size, data in all_results[model_type].items():
            # Get best activations (from largest model)
            activations = data['activations']
            true_labels = data['true_labels']
            
            # Calculate silhouette score if we have at least 2 classes
            if len(np.unique(true_labels)) >= 2:
                # Reduce to 2D for consistency
                pca = PCA(n_components=2)
                reduced_activations = pca.fit_transform(activations)
                
                # Calculate silhouette score
                try:
                    s_score = silhouette_score(reduced_activations, true_labels)
                except:
                    s_score = 0  # Default if calculation fails
                
                model_sizes.append(size)
                silhouette_scores.append(s_score)
        
        # Sort by size
        sorted_indices = np.argsort(model_sizes)
        sorted_sizes = [model_sizes[i] for i in sorted_indices]
        sorted_scores = [silhouette_scores[i] for i in sorted_indices]
        
        # Bar positions for this model type
        x_positions = np.arange(len(sorted_sizes))
        
        # Plot bars
        bar_width = 0.35
        offset = 0 if model_type == 'mlp' else bar_width
        bars = ax3.bar(x_positions + offset, sorted_scores, 
                     bar_width, label=model_type.upper(),
                     color=MODEL_COLORS[model_type], alpha=0.8)
        
        # Add size labels
        for i, bar in enumerate(bars):
            ax3.text(bar.get_x() + bar.get_width()/2, 0.01, 
                   str(sorted_sizes[i]), ha='center', va='bottom',
                   color='white', fontweight='bold', fontsize=10)
    
    # Format plot
    ax3.set_title("Concept Clustering Quality", fontsize=14, color='black')
    ax3.set_xlabel("Model Size Index", fontsize=12, color='black')
    ax3.set_ylabel("Silhouette Score", fontsize=12, color='black')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=12)
    
    # Plot 3: PCA visualizations of latent space
    ax2 = plt.subplot(gs[1, :])
    
    # Create a 2x2 grid within ax2 to show PCA embeddings
    gs_inner = GridSpec(1, len(all_results), 
                       left=ax2.get_position().x0, 
                       right=ax2.get_position().x1,
                       bottom=ax2.get_position().y0,
                       top=ax2.get_position().y1)
    
    # Clear the main subplot
    ax2.set_axis_off()
    
    # For each model type (MLP, CNN)
    for i, model_type in enumerate(all_results.keys()):
        # Find the best (largest) model
        largest_size = max(all_results[model_type].keys())
        data = all_results[model_type][largest_size]
        
        activations = data['activations']
        true_labels = data['true_labels']
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(activations)
        
        # Create subplot
        ax = plt.subplot(gs_inner[i])
        
        # Get unique labels
        unique_labels = np.unique(true_labels)
        
        # Create colormap for classes
        norm = Normalize(vmin=0, vmax=len(unique_labels)-1)
        cmap = plt.cm.viridis
        
        # Scatter each class
        for j, label in enumerate(unique_labels):
            mask = true_labels == label
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                      c=[cmap(norm(j))], 
                      s=100, alpha=0.8, 
                      edgecolor='w', linewidth=0.5)
            
            # Plot class centroid
            centroid = np.mean(embeddings[mask], axis=0)
            ax.scatter(centroid[0], centroid[1], 
                      c=[cmap(norm(j))], 
                      s=200, marker='*', 
                      edgecolor='black', linewidth=1.5,
                      label=f"Class {label}")
            
            # Add text label at centroid
            text = ax.text(centroid[0], centroid[1], f"{label}", 
                          color='white', fontweight='bold', fontsize=10,
                          ha='center', va='center')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                 path_effects.Normal()])
        
        # Format subplot
        ax.set_title(f"{model_type.upper()} (Size: {largest_size})", fontsize=14, color='black')
        ax.set_xlabel("PC 1", fontsize=12, color='black')
        ax.set_ylabel("PC 2", fontsize=12, color='black')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Overall title
    plt.suptitle("Concept Learning: Latent Space Analysis", fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("visualizations/concept_latent_space.png", dpi=300, bbox_inches='tight')
    print("Visualization saved to visualizations/concept_latent_space.png")

def main():
    # Load all activation data
    print("Loading activation data...")
    all_results = load_all_activations()
    
    if not all_results:
        print("No activation data found. Please run concept_learning.py first.")
        return
    
    # Create visualization
    print("Creating latent space visualization...")
    create_latent_space_visualization(all_results)
    
    print("Done!")

if __name__ == "__main__":
    main() 