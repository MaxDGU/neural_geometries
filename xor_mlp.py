import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        np.random.seed(42)  # For reproducibility
        self.hidden_weights = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.output_weights = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.hidden_bias = np.zeros((1, hidden_size))
        self.output_bias = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass
        self.hidden_layer = self.sigmoid(np.dot(X, self.hidden_weights) + self.hidden_bias)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.output_weights) + self.output_bias)
        return self.output_layer
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # Step 1: Compute all errors and deltas
        output_error = y - self.output_layer
        output_delta = output_error * self.sigmoid_derivative(self.output_layer)
        
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)
        
        # Step 2: Compute all weight updates simultaneously
        output_weight_update = learning_rate * np.dot(self.hidden_layer.T, output_delta)
        output_bias_update = learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        hidden_weight_update = learning_rate * np.dot(X.T, hidden_delta)
        hidden_bias_update = learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
        # Step 3: Apply all updates at once
        self.output_weights += output_weight_update
        self.output_bias += output_bias_update
        self.hidden_weights += hidden_weight_update
        self.hidden_bias += hidden_bias_update

class Perceptron:
    def __init__(self, input_size):
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = 0
        
    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def forward(self, X):
        return np.array([[self.step_function(np.dot(x, self.weights) + self.bias)] for x in X])
    
    def train(self, X, y, learning_rate):
        # Rosenblatt's perceptron learning rule
        for i in range(len(X)):
            prediction = self.step_function(np.dot(X[i], self.weights) + self.bias)
            error = y[i][0] - prediction
            self.weights += learning_rate * error * X[i]
            self.bias += learning_rate * error

def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.forward(grid_points)
    predictions = predictions.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, predictions, alpha=0.4)
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=100)
    
    plt.title(title)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_learning_curve(errors, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train_and_evaluate(model, X, y, learning_rate, epochs):
    errors = []
    convergence_epoch = None
    
    for epoch in range(epochs):
        output = model.forward(X)
        
        if isinstance(model, MultiLayerPerceptron):
            model.backward(X, y, learning_rate)
        else:
            model.train(X, y, learning_rate)
        
        error = np.mean(np.square(y - output))
        errors.append(error)
        
        # Check for convergence (error < 0.01)
        if convergence_epoch is None and error < 0.01:
            convergence_epoch = epoch
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}, Error: {error:.4f}")
    
    return errors, convergence_epoch

def plot_performance_comparison(results):
    # Plot convergence epochs
    plt.figure(figsize=(10, 6))
    hidden_sizes = list(results.keys())
    convergence_epochs = [results[size]['convergence_epoch'] for size in hidden_sizes]
    plt.plot(hidden_sizes, convergence_epochs, 'b-o')
    plt.title('Convergence Speed vs Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Epochs to Convergence')
    plt.grid(True)
    plt.savefig('convergence_comparison.png')
    plt.close()
    
    # Plot final errors
    plt.figure(figsize=(10, 6))
    final_errors = [results[size]['final_error'] for size in hidden_sizes]
    plt.plot(hidden_sizes, final_errors, 'r-o')
    plt.title('Final Error vs Hidden Layer Size')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Final Mean Squared Error')
    plt.grid(True)
    plt.savefig('error_comparison.png')
    plt.close()

def main():
    # XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Training parameters
    learning_rate = 0.1
    epochs = 10000
    
    # Test different hidden layer sizes
    hidden_sizes = [2, 4, 8, 16, 32]
    results = {}
    
    for hidden_size in hidden_sizes:
        print(f"\nTesting hidden layer size: {hidden_size}")
        model = MultiLayerPerceptron(2, hidden_size, 1)
        errors, convergence_epoch = train_and_evaluate(model, X, y, learning_rate, epochs)
        
        # Store results
        results[hidden_size] = {
            'errors': errors,
            'convergence_epoch': convergence_epoch if convergence_epoch is not None else epochs,
            'final_error': errors[-1]
        }
        
        # Save plots for this configuration
        plot_decision_boundary(model, X, y, 
                             f'Decision Boundary - Hidden Size {hidden_size}',
                             f'decision_boundary_hidden_{hidden_size}.png')
        plot_learning_curve(errors,
                          f'Learning Curve - Hidden Size {hidden_size}',
                          f'learning_curve_hidden_{hidden_size}.png')
    
    # Plot comparison of different hidden layer sizes
    plot_performance_comparison(results)
    
    # Print summary
    print("\nPerformance Summary:")
    for hidden_size in hidden_sizes:
        result = results[hidden_size]
        print(f"Hidden Size {hidden_size}:")
        print(f"  Final Error: {result['final_error']:.4f}")
        print(f"  Epochs to Convergence: {result['convergence_epoch']}")

if __name__ == "__main__":
    main() 