# Introduction to TensorFlow

## 📌 Metadata
- **Category:** Foundations
- **Primary Domain:** Deep Learning, ML Frameworks
- **Difficulty:** Beginner
- **Prerequisites:** Python basics, NumPy arrays, Basic linear algebra (vectors/matrices), Neural network concepts (nodes, layers, loss)

## 📋 Summary
- **What It Does:** TensorFlow is an open-source numerical computation library by Google that lets you define, train, and deploy machine learning models using computational graphs and automatic differentiation.
- **When to Use:** When building and training neural networks, especially when you need GPU acceleration, production deployment (TensorFlow Serving, TFLite), or scalable distributed training.
- **When Not to Use:** When your task is purely statistical (use `scikit-learn`), when you need fast prototyping with maximum flexibility (consider PyTorch), or when the model is trivially small and a full framework is overkill.

## 📖 Definitions
- **Input:** Raw data (images, text, numbers) represented as multi-dimensional arrays called **tensors**.
- **Output:** Predictions, class probabilities, or regression values depending on the task.
- **Invariant / Assumption:** Data must be numeric and shaped consistently; all tensors in a batch must share the same dimensions except the batch axis.

### Key Vocabulary
| Term | Meaning |
|---|---|
| **Tensor** | An n-dimensional array; the fundamental data unit in TensorFlow |
| **Model** | A function with learned parameters that maps inputs to outputs |
| **Layer** | A single transformation stage inside a model (e.g., Dense, Conv2D) |
| **Loss Function** | Measures how wrong the model's predictions are |
| **Optimizer** | Algorithm that updates weights to reduce loss (e.g., Adam, SGD) |
| **Epoch** | One full pass through the entire training dataset |
| **Gradient** | Direction and magnitude of the steepest loss increase; used in reverse to update weights |
| **Backpropagation** | Algorithm that computes gradients layer by layer using the chain rule |

## ⚙️ Workflow (Training a Neural Network)
1. **Initialize:** Define the model architecture using `tf.keras.Sequential` or the functional API. Set layer types, sizes, and activation functions.
2. **Compile:** Attach a loss function, optimizer, and metrics via `model.compile(...)`.
3. **Process:** Feed batches of training data forward through the model (forward pass) to produce predictions.
4. **Update:** Compute loss, then run backpropagation to calculate gradients. The optimizer adjusts weights in the direction that reduces loss.
5. **Terminate:** Repeat steps 3–4 for a fixed number of epochs, or until validation loss stops improving (early stopping).
6. **Return:** A trained model that can be called on new data via `model.predict(...)`.

### Visual Flow
```
[Raw Data]
    ↓
[Preprocessing / Normalization]
    ↓
[Model: Input → Hidden Layers → Output]
    ↓
[Loss Computation]
    ↓
[Backpropagation → Gradient Updates]
    ↓
[Repeat per Epoch]
    ↓
[Trained Model → Predictions]
```

## 🧮 Complexity Analysis

> Note: Complexity here refers to a single forward + backward pass through a **Dense (fully connected)** layer.

- **Time Complexity:**
  - **Best Case (Ω):** $\Omega(n \cdot d)$ — for a single layer with $n$ inputs and $d$ neurons
  - **Average Case (Θ):** $\Theta(L \cdot n \cdot d)$ — for $L$ layers, $n$ inputs per layer, $d$ neurons per layer
  - **Worst Case (O):** $O(E \cdot B \cdot L \cdot n \cdot d)$ — full training with $E$ epochs, batch size $B$

- **Space Complexity:** $O(L \cdot n \cdot d)$ for storing weights; $O(B \cdot n)$ for activations per batch.

### Complexity Table (Dense Layer, Approximate)
| Scenario | Operations (Approx.) | Notes |
|---|---|---|
| 1 layer, 64 neurons, 1 sample | ~$10^3$–$10^4$ | Fast; negligible on CPU |
| 3 layers, 256 neurons, batch 32 | ~$10^6$–$10^7$ | Comfortable for GPU |
| 10+ layers, 1024 neurons, batch 128 | ~$10^9$+ | Requires GPU; tuning needed |

## 🔁 Correctness Intuition
- TensorFlow uses **automatic differentiation** (`tf.GradientTape`) to compute exact gradients without manually deriving them. This guarantees numerically correct gradient updates as long as operations are differentiable.
- The model converges because each weight update moves the parameters in the direction that locally reduces the loss function. Given a well-chosen learning rate, gradient descent finds a minimum (local or global).
- Keras's `model.fit()` handles shuffling, batching, and metric aggregation internally, reducing sources of implementation error.

## ⚖️ Tradeoffs
- **Strengths:**
  - Production-ready with TensorFlow Serving, TFLite, and TensorFlow.js for deployment
  - Strong GPU/TPU support via CUDA
  - Keras high-level API makes model building fast and readable
  - Large ecosystem: TensorFlow Hub, TensorFlow Datasets, TensorBoard
- **Weaknesses:**
  - Steeper learning curve than PyTorch for research-style experimentation
  - Eager execution (default in TF2) is more intuitive but can be slower than graph mode for large-scale production
  - Debugging raw tensor operations is harder than standard Python debugging
- **Alternatives:** PyTorch (more popular in research), JAX (functional, XLA-compiled), scikit-learn (classical ML, no deep learning)

## 💻 Implementation (Python)

### Example 1 — Basic Tensor Operations
```python
import tensorflow as tf

# Creating tensors
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
print("Add:\n", tf.add(a, b).numpy())
print("Multiply:\n", tf.multiply(a, b).numpy())
print("Matrix multiply:\n", tf.matmul(a, b).numpy())
```

### Example 2 — Binary Classifier on Synthetic Data
```python
import tensorflow as tf
import numpy as np

# ── 1. Synthetic dataset ──────────────────────────────────────────────────────
np.random.seed(42)
X = np.random.randn(200, 2).astype(np.float32)
# Label: 1 if point is above the line y = x, else 0
y = (X[:, 1] > X[:, 0]).astype(np.float32)

# ── 2. Model definition ───────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),   # Binary output
])

# ── 3. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
history = model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2, verbose=0)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Final Loss:     {loss:.4f}")
print(f"Final Accuracy: {accuracy:.4f}")

# ── 6. Predict ────────────────────────────────────────────────────────────────
sample = np.array([[0.5, 1.5], [1.5, 0.5]], dtype=np.float32)
predictions = model.predict(sample, verbose=0)
print(f"\nPredictions (probability class=1): {predictions.flatten()}")
```

### Example 3 — Manual Gradient with GradientTape
```python
import tensorflow as tf

# Simple function: f(x) = x^2 + 3x + 2
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2 + 3*x + 2          # Forward pass

grad = tape.gradient(y, x)      # df/dx = 2x + 3
print(f"f(3)  = {y.numpy()}")   # Expected: 20
print(f"f'(3) = {grad.numpy()}") # Expected: 9
```

## 🧪 Test Cases
| Case | Input | Expected Output | Purpose |
|---|---|---|---|
| 1 | Two linearly separable 2D points | Accuracy ≥ 0.90 after 30 epochs | Basic training sanity check |
| 2 | Point `[1.5, 0.5]` (below y=x) | Prediction close to 0.0 | Correct class-0 inference |
| 3 | Point `[0.5, 1.5]` (above y=x) | Prediction close to 1.0 | Correct class-1 inference |
| 4 | GradientTape on $f(x)=x^2+3x+2$ at $x=3$ | Gradient = 9.0 | Automatic differentiation correctness |
| 5 | All-zero input `[[0, 0]]` | Prediction near 0.5 (ambiguous boundary) | Boundary/edge case behavior |

## 📚 References
- [TensorFlow Official Documentation](https://www.tensorflow.org/api_docs/python/tf) — Full API reference
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras) — High-level model building
- [MIT 6.S191 — Introduction to Deep Learning](http://introtodeeplearning.com/) — Lecture slides and labs using TensorFlow
- [Hands-On Machine Learning (Aurélien Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Practical TF2 + Keras coverage
- [TensorFlow Playground](https://playground.tensorflow.org/) — Visual neural network sandbox, no code required