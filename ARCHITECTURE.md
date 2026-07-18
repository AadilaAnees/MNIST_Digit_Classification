# Architecture & Technical Deep-Dive

## 1. Data Pipeline

**Ingestion:** Raw idx-ubyte files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) parsed with `idx2numpy.convert_from_file()`. These files use a custom binary header format (magic number + dimension metadata) followed by raw pixel/label bytes — parsing them directly (rather than using `keras.datasets.mnist`) was a deliberate choice to practice handling non-trivial real-world data formats.

**Preprocessing steps, in order:**
1. **Reshape** — images reshaped from `(N, 28, 28)` to `(N, 28, 28, 1)` to add an explicit channel dimension, since `Conv2D` layers require a 4D input `(samples, height, width, channels)`.
2. **Normalization** — pixel values cast to `float32` and divided by 255.0, scaling the range from [0, 255] to [0, 1]. This produces more stable gradients during backpropagation.
3. **One-hot encoding** — integer labels (0–9) converted to 10-length one-hot vectors via `keras.utils.to_categorical()`, matching the softmax output layer and `categorical_crossentropy` loss.

**Final tensor shapes:**
- Train images: `(60000, 28, 28, 1)`
- Test images: `(10000, 28, 28, 1)`
- Train labels: `(60000, 10)`
- Test labels: `(10000, 10)`

## 2. Model Architecture

```
Input (28, 28, 1)
  → Conv2D(32 filters, 3x3, ReLU)
  → MaxPooling2D(2x2)
  → Conv2D(64 filters, 3x3, ReLU)
  → MaxPooling2D(2x2)
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.5)
  → Dense(10, Softmax)
```

**Design rationale:**
- **Two Conv+Pool blocks:** the first block detects low-level features (edges, strokes); the second combines these into higher-level shapes (loops, curves) relevant to distinguishing digits like 0/6/8/9.
- **ReLU activation:** computationally cheap, avoids vanishing gradients that older activations (sigmoid/tanh) suffer from in deeper stacks.
- **MaxPooling:** reduces spatial dimensions (down-samples), cutting compute cost and making the network robust to small positional shifts of the digit within the frame.
- **Dropout(0.5):** randomly disables 50% of neurons in the dense layer during training, forcing the network to not over-rely on specific neurons — reduces overfitting risk. Note: this is a relatively aggressive dropout rate for a dataset as clean as MNIST, which is part of why training accuracy visibly lagged behind validation accuracy during training (dropout is active in training mode, inactive during validation/inference).
- **Softmax output:** converts final layer outputs into a probability distribution across the 10 digit classes, summing to 1.

## 3. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |
| Metric tracked | Accuracy |
| Epochs | 10 |
| Batch size | 64 *(inferred from training log step count — confirm against your actual code)* |
| Validation split | 0.1 *(inferred — confirm against your actual code)* |

**Training behavior observed:** both training and validation accuracy climbed steadily across all 10 epochs with no divergence (validation loss kept decreasing alongside training loss) — the signature of a well-generalizing model, not an overfit one. Validation accuracy consistently ran ahead of training accuracy throughout, which is expected given Dropout is active only during training.

## 4. Evaluation Methodology

- **Held-out test set:** evaluation performed on the 10,000-image `t10k` files, which were never used during training or validation-split monitoring — this is the unbiased, final performance measure.
- **Classification report:** per-class precision, recall, and F1-score computed via `sklearn.metrics.classification_report`, using `np.argmax()` to convert both softmax outputs and one-hot labels back to plain digit predictions before comparison.
- **Confusion matrix:** 10×10 true-vs-predicted grid via `sklearn.metrics.confusion_matrix`, visualized as a Seaborn heatmap to make misclassification clusters visually obvious.

### Results Summary

| Metric | Score |
|---|---|
| Test Accuracy | 97.76% |
| Test Loss | 0.0716 |
| Misclassified | 224 / 10,000 |

### Per-Class Performance

| Digit | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.97 | 0.99 | 0.98 | 980 |
| 1 | 0.99 | 0.99 | 0.99 | 1135 |
| 2 | 0.98 | 0.98 | 0.98 | 1032 |
| 3 | 0.98 | 0.97 | 0.98 | 1010 |
| 4 | 0.98 | 0.98 | 0.98 | 982 |
| 5 | 0.98 | 0.98 | 0.98 | 892 |
| 6 | 0.98 | 0.99 | 0.98 | 958 |
| 7 | 0.97 | 0.97 | 0.97 | 1028 |
| 8 | 0.98 | 0.95 | 0.97 | 974 |
| 9 | 0.97 | 0.96 | 0.97 | 1009 |

**Notable error pattern:** digits 8 and 9 show the lowest recall (0.95, 0.96 respectively), and misclassifications such as true-5-predicted-as-7 were observed directly — both consistent with well-documented MNIST confusion pairs caused by visually similar stroke patterns in certain handwriting styles.

## 5. Production Considerations

- **Model persistence:** trained model saved via `model.save('mnist_cnn.keras')`, allowing it to be reloaded (`keras.models.load_model()`) without retraining — verified by reloading and re-evaluating to confirm identical test accuracy.
- **Reusable functions:** data loading and model construction wrapped into standalone functions (`load_mnist_data()`, `build_cnn_model()`) to decouple the pipeline from a single linear notebook script, making it easier to reuse or extend.

## 6. Possible Future Improvements

- Data augmentation (small rotations/shifts) to further improve robustness on edge-case handwriting styles.
- Tuning dropout rate (e.g. 0.25 instead of 0.5) given MNIST's relative simplicity.
- Batch normalization layers to potentially speed up convergence.
- Deploying the saved model behind a small FastAPI or Streamlit interface for live interactive digit-drawing predictions.
