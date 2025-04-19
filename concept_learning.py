import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from itertools import product
import os

# Create directories for saving results
os.makedirs('results/activations', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

class ConceptGenerator:
    def __init__(self):
        self.shapes = ['triangle', 'square', 'lines']
        self.colors = ['blue', 'red']
        self.patterns = ['solid', 'striped', 'hatched']
        
        # Define probabilities for PCFG
        self.shape_probs = [0.33, 0.33, 0.34]
        self.color_probs = [0.5, 0.5]
        self.pattern_probs = [0.33, 0.33, 0.34]
        
    def generate_concept(self):
        shape = np.random.choice(self.shapes, p=self.shape_probs)
        color = np.random.choice(self.colors, p=self.color_probs)
        pattern = np.random.choice(self.patterns, p=self.pattern_probs)
        return shape, color, pattern
    
    def concept_to_bitstring(self, shape, color, pattern):
        # Encode concept as 4-bit string (as shown in image)
        shape_idx = self.shapes.index(shape)
        color_idx = self.colors.index(color)
        pattern_idx = self.patterns.index(pattern)
        
        # First two bits for shape, third for color, fourth for pattern
        bitstring = np.zeros(4, dtype=np.float32)
        bitstring[0] = shape_idx // 2
        bitstring[1] = shape_idx % 2
        bitstring[2] = color_idx
        bitstring[3] = pattern_idx // 2
        return bitstring
    
    def concept_to_image(self, shape, color, pattern):
        # Generate 32x32 image representation with RGB channels
        img = np.zeros((32, 32, 3), dtype=np.float32)
        
        # Set color values
        color_map = {
            'blue': [0.0, 0.0, 1.0],
            'red': [1.0, 0.0, 0.0]
        }
        rgb = color_map[color]
        
        if shape == 'triangle':
            # Draw triangle - centered and larger
            for i in range(32):
                width = min(i, 32-i)  # Makes a more symmetric triangle
                for j in range(16-width, 16+width):
                    img[i, j] = rgb
        elif shape == 'square':
            # Draw square - centered
            margin = 8
            img[margin:32-margin, margin:32-margin] = rgb
        else:  # lines
            # Draw vertical lines - more visible
            for i in range(4, 28, 6):  # Adjusted spacing
                img[:, i:i+4] = rgb  # Thicker lines
        
        # Apply pattern
        if pattern == 'striped':
            # Horizontal stripes
            mask = np.zeros((32, 32, 3))
            mask[::3] = 1  # Wider stripes (every 3rd row)
            img *= mask
        elif pattern == 'hatched':
            # Diagonal hatching
            mask = np.zeros((32, 32, 3))
            for i in range(32):
                for j in range(32):
                    if (i + j) % 4 == 0:  # Diagonal lines every 4 pixels
                        mask[i, max(0, j-1):min(32, j+2)] = 1  # Thicker lines
            img *= mask
        
        return img

    def visualize_concepts(self, num_examples=5):
        """Visualize random examples of generated concepts"""
        plt.figure(figsize=(15, 6))
        
        # Generate some random concepts
        for i in range(num_examples):
            shape, color, pattern = self.generate_concept()
            
            # Get both representations
            img = self.concept_to_image(shape, color, pattern)
            bits = self.concept_to_bitstring(shape, color, pattern)
            
            # Plot image
            plt.subplot(2, num_examples, i + 1)
            plt.imshow(img)  # Remove cmap='gray' to show colors
            plt.title(f'{color} {shape}\n{pattern}')
            plt.axis('off')
            
            # Plot bitstring representation
            plt.subplot(2, num_examples, num_examples + i + 1)
            plt.bar(range(4), bits)
            plt.ylim(-0.1, 1.1)
            plt.title('Bitstring')
            plt.xticks(range(4), ['s1', 's2', 'color', 'pattern'])
        
        plt.tight_layout()
        plt.savefig('concept_examples.png')
        plt.close()

class ConceptDataset(Dataset):
    def __init__(self, num_samples, input_type='bitstring'):
        self.generator = ConceptGenerator()
        self.input_type = input_type
        self.num_samples = num_samples
        
        # Generate dataset
        self.data = []
        self.labels = []
        self.generate_data()
        
    def generate_data(self):
        for _ in range(self.num_samples):
            shape, color, pattern = self.generator.generate_concept()
            
            # Generate input representation
            if self.input_type == 'bitstring':
                x = self.generator.concept_to_bitstring(shape, color, pattern)
            else:  # image
                x = self.generator.concept_to_image(shape, color, pattern)
                # For CNN input, we want shape (H, W, C)
                x = x.astype(np.float32)
            
            # Generate label (unique category for each concept combination)
            label = (self.generator.shapes.index(shape) * len(self.generator.colors) * len(self.generator.patterns) +
                    self.generator.colors.index(color) * len(self.generator.patterns) +
                    self.generator.patterns.index(pattern))
            
            self.data.append(x)
            self.labels.append(label)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])[0]
        return x, y

class MLP(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.output_layer = nn.Linear(hidden_size, 18)  # 18 possible concepts
    
    def forward(self, x, return_activations=False):
        activations = self.hidden_layer(x)
        output = self.output_layer(activations)
        
        if return_activations:
            return output, activations
        return output

class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        
        # Start with 3 channels (RGB), double channels in each layer
        channels = [3] + [16 * (2**i) for i in range(num_conv_layers)]
        
        # Create convolutional layers with BatchNorm and Dropout
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Sequential(
                # First conv block
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                # Second conv block for more non-linearity
                nn.Conv2d(channels[i+1], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(),
                # Pooling and dropout
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate)
            ))
        
        # Calculate size of flattened features after convolutions
        feature_size = 32 // (2**num_conv_layers)  # Due to pooling layers
        flat_features = channels[-1] * feature_size * feature_size
        
        # Feature extraction layer (before final classification)
        self.feature_extractor = nn.Sequential(
            nn.Linear(flat_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(128, 18)  # 18 possible concepts
        
    def forward(self, x, return_activations=False):
        # Debug print to understand input shape
        input_shape = x.shape
        
        # For image data
        if len(x.shape) == 4:  # Batch of images with shape (B, H, W, 3)
            x = x.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        elif len(x.shape) == 3:  # Single image with shape (H, W, 3)
            if x.shape[-1] == 3:  # Make sure the last dimension is RGB
                x = x.permute(2, 0, 1).unsqueeze(0)  # Change to (1, C, H, W)
            else:
                # If channels are already in the first dimension
                x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
        
        # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(x.size(0), -1)
        
        # Get features before classification
        features = self.feature_extractor(x)
        
        # Final classification
        output = self.classifier(features)
        
        if return_activations:
            return output, features
        return output

def visualize_feature_maps(model, sample_image, layer_idx):
    """Visualize feature maps from a specific convolutional layer"""
    model.eval()
    
    # Forward pass until the specified layer
    with torch.no_grad():
        # Get input in the right format (B, C, H, W)
        if isinstance(sample_image, np.ndarray):
            if len(sample_image.shape) == 3 and sample_image.shape[2] == 3:
                # Image is (H, W, C)
                x = torch.FloatTensor(sample_image).unsqueeze(0)  # Add batch dim -> (1, H, W, C)
                x = x.permute(0, 3, 1, 2)  # Change to (1, C, H, W)
            else:
                x = torch.FloatTensor(sample_image).unsqueeze(0)
        else:
            # Already a tensor
            x = sample_image.unsqueeze(0) if len(sample_image.shape) == 3 else sample_image
            
            # Ensure it's in (B, C, H, W) format
            if x.shape[-1] == 3:  # If channels are last
                x = x.permute(0, 3, 1, 2)
                
        # Apply layers up to the specified one
        for i, layer in enumerate(model.conv_layers):
            x = layer(x)
            if i == layer_idx:
                break
    
    # Plot feature maps
    feature_maps = x.squeeze().detach().cpu().numpy()
    num_maps = min(16, feature_maps.shape[0])  # Show at most 16 feature maps
    
    plt.figure(figsize=(15, 15))
    for i in range(num_maps):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.suptitle(f'Feature Maps - Layer {layer_idx + 1}')
    plt.tight_layout()
    plt.savefig(f'visualizations/feature_maps_layer_{layer_idx+1}.png')
    plt.close()

def train_model(model, train_loader, test_loader, device, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # For debugging the first batch
            if epoch == 0 and batch_idx == 0:
                print(f"Input data shape: {x.shape}")
                if isinstance(model, SimpleCNN):
                    print("Training CNN - checking input format")
                else:
                    print("Training MLP - checking input format")
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                test_loss += criterion(output, y).item()
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        test_accuracies.append(accuracy)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:  # Stop if no improvement for 15 epochs
                print("Early stopping triggered")
                break
    
    return train_losses, test_accuracies

def evaluate_and_save_activations(model, data_loader, device, model_type, model_size):
    """Evaluate model and save activations of the last layer."""
    model.eval()
    all_activations = []
    all_labels = []
    all_raw_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs, activations = model(x, return_activations=True)
            
            # Save activations and labels
            all_activations.append(activations.cpu().numpy())
            
            # Save original concept labels
            all_raw_labels.append(y.cpu().numpy())
            
            # Predicted labels (for comparison)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.append(predicted.cpu().numpy())
    
    # Concatenate all batches
    all_activations = np.concatenate(all_activations, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_raw_labels = np.concatenate(all_raw_labels, axis=0)
    
    # Save to file
    np.savez(
        f'results/activations/{model_type}_{model_size}.npz',
        activations=all_activations,
        predicted_labels=all_labels,
        true_labels=all_raw_labels
    )
    
    return all_activations, all_raw_labels

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Visualize some example concepts
    generator = ConceptGenerator()
    generator.visualize_concepts()
    
    # Parameters
    hidden_sizes = [16, 32, 64, 128, 256]  # For MLP
    conv_layers = [1, 2, 3, 4]  # Number of conv layers for CNN
    num_train = 1000
    num_test = 200
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Results storage
    mlp_results = {}
    cnn_results = {}
    
    # Create datasets for later use
    test_image_dataset = ConceptDataset(1, 'image')
    sample_image = test_image_dataset.data[0]
    
    # Train and evaluate MLPs
    print("\nTraining MLPs with different hidden sizes...")
    for hidden_size in hidden_sizes:
        print(f"\nHidden size: {hidden_size}")
        
        # Create datasets
        train_dataset = ConceptDataset(num_train, 'bitstring')
        test_dataset = ConceptDataset(num_test, 'bitstring')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Train model
        model = MLP(hidden_size).to(device)
        train_losses, test_accuracies = train_model(model, train_loader, test_loader, device)
        
        mlp_results[hidden_size] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
        
        # Save activations
        print(f"Saving activations for MLP with hidden size {hidden_size}...")
        evaluate_and_save_activations(model, test_loader, device, 'mlp', hidden_size)
    
    # Train and evaluate CNNs
    print("\nTraining CNNs with different numbers of convolutional layers...")
    deepest_cnn = None
    
    for num_layers in conv_layers:
        print(f"\nNumber of convolutional layers: {num_layers}")
        
        # Create datasets
        train_dataset = ConceptDataset(num_train, 'image')
        test_dataset = ConceptDataset(num_test, 'image')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Train model
        model = SimpleCNN(num_layers).to(device)
        train_losses, test_accuracies = train_model(model, train_loader, test_loader, device)
        
        cnn_results[num_layers] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
        
        # Save activations
        print(f"Saving activations for CNN with {num_layers} layers...")
        evaluate_and_save_activations(model, test_loader, device, 'cnn', num_layers)
        
        # Save the deepest CNN for feature map visualization
        if num_layers == max(conv_layers):
            deepest_cnn = model
    
    # Visualize feature maps from each layer of the deepest CNN
    if deepest_cnn is not None:
        print("\nVisualizing feature maps...")
        for layer_idx in range(max(conv_layers)):
            visualize_feature_maps(deepest_cnn, sample_image, layer_idx)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot MLP results
    plt.subplot(2, 2, 1)
    plt.plot(hidden_sizes, [results['test_accuracies'][-1] for results in mlp_results.values()], 'b-o')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Final Test Accuracy')
    plt.title('MLP Performance vs Hidden Layer Size')
    plt.grid(True)
    
    # Plot CNN results
    plt.subplot(2, 2, 2)
    plt.plot(conv_layers, [results['test_accuracies'][-1] for results in cnn_results.values()], 'r-o')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Final Test Accuracy')
    plt.title('CNN Performance vs Network Depth')
    plt.grid(True)
    
    # Plot MLP learning curves
    plt.subplot(2, 2, 3)
    plt.plot(mlp_results[hidden_sizes[-1]]['train_losses'], 'b-', label=f'Hidden Size={hidden_sizes[-1]}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('MLP Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot CNN learning curves
    plt.subplot(2, 2, 4)
    for num_layers in conv_layers:
        plt.plot(cnn_results[num_layers]['train_losses'], 
                label=f'{num_layers} layers', 
                alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('CNN Learning Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/concept_learning_results.png')
    plt.close()
    
    # Print summary
    print("\nPerformance Summary:")
    print("\nMLP Results:")
    for hidden_size in hidden_sizes:
        print(f"Hidden Size {hidden_size}: {mlp_results[hidden_size]['test_accuracies'][-1]:.4f}")
    
    print("\nCNN Results:")
    for num_layers in conv_layers:
        print(f"{num_layers} Conv Layers: {cnn_results[num_layers]['test_accuracies'][-1]:.4f}")

if __name__ == "__main__":
    main() 