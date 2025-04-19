# Meta-Learning Experiment Results

This document summarizes the results of comparing vanilla and meta-trained models on concept learning tasks.

## Experiment Overview

We implemented and compared two learning approaches:
1. **Vanilla Learning**: Standard supervised learning on concept classification tasks
2. **Meta-Learning**: Using Model-Agnostic Meta-Learning (MAML) from the learn2learn library

The experiment was conducted on three model architectures:
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- Transformer 

## Training Process

### Dataset
- Synthetic concept learning tasks were generated using the MetaBitConceptsDataset from ManyPaths
- Each task consisted of support (training) and query (test) sets
- For CNN models, feature vectors were converted to image representations
- For transformer models, bit vector data was used with appropriate sequence formatting

### Training Configuration
- **Vanilla Models**: Trained using standard gradient descent with BCE loss
- **Meta-Trained Models**: Trained using MAML with inner and outer optimization loops
- Both approaches used Adam optimizer

## Weight Space Visualization

The visualizations in `visualizations/weight_space/` show the weight space trajectories of vanilla and meta-trained models:

### MLP Model
![MLP Weight Space](visualizations/weight_space/mlp_weight_space_comparison.png)

### CNN Model
![CNN Weight Space](visualizations/weight_space/cnn_weight_space_comparison.png)

### Transformer Model
![Transformer Weight Space](visualizations/weight_space/transformer_weight_space_comparison.png)

## Key Observations

1. **Initialization Point**: Meta-trained models have different initialization points in weight space compared to vanilla models, suggesting the meta-learning algorithm finds parameter configurations that are better suited for quick adaptation.

2. **Adaptation Trajectories**: 
   - Vanilla models show more varied trajectories during adaptation
   - Meta-trained models have more consistent adaptation patterns

3. **Final Configurations**: Meta-trained models adapt to reach similar regions in weight space, indicating they converge to similar solutions across different tasks.

4. **Loss Patterns**:
   - Vanilla training shows steady decreases in loss
   - Meta-training losses are relatively stable, suggesting the model focuses on finding good initialization rather than direct task performance

5. **Model Architecture Differences**:
   - Transformers showed more complex weight space dynamics compared to MLP and CNN models
   - The transformer's attention mechanisms resulted in different adaptation patterns
   - Meta-learning was particularly effective for transformer models, demonstrating greater convergence in weight space after adaptation

## Conclusion

The weight space visualization reveals the fundamental difference between standard learning and meta-learning:

- **Vanilla learning** optimizes for performance on a specific task
- **Meta-learning** optimizes for quick adaptation across many tasks

The MAML approach successfully finds model parameters that can adapt quickly to new tasks with minimal gradient updates, demonstrated by the more consistent trajectories in weight space after adaptation.

This experiment validates that meta-learning is learning a "learning strategy" rather than just task-specific solutions, and this effect is consistent across different neural network architectures, from simple MLPs to complex transformers. 