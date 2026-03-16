# Supervised Learning

## 📌 Metadata
- **Category:** Foundations
- **Primary Domain:** Machine Learning / AI
- **Difficulty:** Intermediate
- **Prerequisites:** Linear algebra, Probability & statistics, Python (NumPy/scikit-learn), Gradient descent

---

## 📋 Summary
- **What It Does:** Supervised learning trains a model on labeled input-output pairs so it can predict outputs for unseen inputs by learning the mapping $f: X \rightarrow Y$.
- **When to Use:** When labeled training data is available and the goal is to predict a discrete class (classification) or a continuous value (regression).
- **When Not to Use:** When labels are unavailable or too expensive to obtain — use unsupervised or self-supervised learning instead. Also avoid when the input distribution shifts heavily between training and deployment (distribution shift problem).

---

## 📖 Definitions
- **Input ($X$):** A dataset of $n$ samples, each with $d$ features: $X \in \mathbb{R}^{n \times d}$
- **Output ($Y$):** Ground-truth labels — either class indices $y \in \{0, 1, \ldots, k-1\}$ (classification) or real values $y \in \mathbb{R}$ (regression)
- **Model ($f_\theta$):** A parameterized function $f_\theta: X \rightarrow \hat{Y}$ that produces predictions
- **Loss Function ($\mathcal{L}$):** Measures the error between predictions $\hat{Y}$ and true labels $Y$
- **Invariant / Assumption:** The training and test data are drawn i.i.d. (independently and identically distributed) from the same underlying distribution $P(X, Y)$

---

## ⚙️ Workflow
1. **Initialize:** Define a model architecture $f_\theta$ and randomly initialize parameters $\theta$. Split data into training, validation, and test sets.
2. **Process (Forward Pass):** For each batch of inputs $X_b$, compute predictions $\hat{Y}_b = f_\theta(X_b)$.
3. **Update (Backward Pass):** Compute the loss $\mathcal{L}(\hat{Y}_b, Y_b)$, compute gradients $\nabla_\theta \mathcal{L}$, and update parameters: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, where $\eta$ is the learning rate.
4. **Terminate:** Stop when validation loss stops improving (early stopping) or after a fixed number of epochs.
5. **Return:** The trained model $f_\theta^*$ and its evaluation metrics on the held-out test set.

---

## 🧮 Complexity Analysis

> Complexity depends on the specific model. The values below apply to a general **linear model** (Logistic Regression / Linear Regression) as a reference baseline.

- **Time Complexity:**
  - **Best Case ($\Omega$):** $\Omega(nd)$ — one pass through all $n$ samples with $d$ features
  - **Average Case ($\Theta$):** $\Theta(nd \cdot E)$ — $E$ epochs of gradient descent over all samples
  - **Worst Case ($O$):** $O(nd \cdot E)$ — same, assuming full-batch gradient descent
- **Space Complexity:** $O(nd + d)$ — store the dataset and parameter vector

### Complexity Table
| Input Size | Operations (Approx.) | Notes |
|---|---|---|
| Small ($n = 10^3$, $d = 10$) | $\sim 10^4$ per epoch | Fast; any model works |
| Medium ($n = 10^5$, $d = 100$) | $\sim 10^7$ per epoch | Mini-batch SGD preferred |
| Large ($n = 10^7$, $d = 10^3$) | $\sim 10^{10}$ per epoch | Distributed training or approximate methods needed |

---

## 🔁 Correctness Intuition
- The model converges because gradient descent iteratively reduces the loss function toward a (local) minimum, provided the learning rate is appropriately small.
- Generalization is justified by the i.i.d. assumption — if training and test data share the same distribution, minimizing training loss approximates minimizing true expected loss (generalization error).
- Regularization (L1/L2) prevents overfitting by penalizing large parameter values, keeping the model's hypothesis class simpler.

---

## ⚖️ Tradeoffs
- **Strengths:**
  - Directly optimizes for the target task when labels are available.
  - Well-understood theoretical guarantees (PAC learning, VC dimension).
  - Wide range of models available — from simple (linear) to complex (deep networks).
- **Weaknesses:**
  - Requires labeled data, which is often expensive to collect.
  - Sensitive to label noise and class imbalance.
  - Can overfit when the model is too complex relative to the training set size.
  - Poor performance when test distribution differs from training distribution.
- **Alternatives:**
  - Unsupervised Learning (clustering, autoencoders) — when labels are unavailable
  - Semi-supervised Learning — when only a small subset of data is labeled
  - Self-supervised Learning — when labels can be derived from the data itself (e.g., masked language modeling)
  - Reinforcement Learning — when feedback comes as rewards, not fixed labels

---

## 💻 Implementation (Python)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def supervised_learning_pipeline(X, y, test_size=0.2, random_state=42):
    """
    A minimal supervised learning pipeline for binary or multi-class classification.

    Assumptions:
    - X is a 2D NumPy array of shape (n_samples, n_features).
    - y is a 1D NumPy array of integer class labels.
    - Data is i.i.d. sampled from a fixed distribution.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_classes=2,
        random_state=42
    )

    model, accuracy, report = supervised_learning_pipeline(X, y)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
```

---

## 🧪 Test Cases
| Case | Input | Expected Output | Purpose |
|---|---|---|---|
| 1 | 1000 samples, 10 features, 2 classes (linearly separable) | Accuracy ≥ 0.90 | Basic sanity — model learns a clean boundary |
| 2 | 100 samples, 20 features (underdetermined, $n \ll d$) | Accuracy degrades; near 0.50–0.60 | Overfitting / curse of dimensionality |
| 3 | All samples from one class ($y = [0, 0, \ldots, 0]$) | Model outputs 0 for all; accuracy = 1.0 but useless | Class imbalance edge case |
| 4 | Features with no correlation to labels (pure noise) | Accuracy ≈ 0.50 (random chance) | Confirms model does not hallucinate signal |

---

## 🗂️ Key Supervised Learning Models (Reference)

| Model | Task | Key Hyperparameters | Notes |
|---|---|---|---|
| Logistic Regression | Classification | `C` (regularization), `max_iter` | Linear decision boundary |
| Linear Regression | Regression | `fit_intercept` | Assumes linear relationship |
| Decision Tree | Both | `max_depth`, `min_samples_split` | Prone to overfitting |
| Random Forest | Both | `n_estimators`, `max_depth` | Ensemble; reduces variance |
| SVM | Both | `C`, `kernel`, `gamma` | Strong on small, high-dim data |
| k-NN | Both | `k`, distance metric | Lazy learner; slow at inference |
| Neural Network | Both | layers, `lr`, `batch_size` | Most flexible; needs most data |

---

## 📐 Common Loss Functions

| Task | Loss Function | Formula |
|---|---|---|
| Binary Classification | Binary Cross-Entropy | $-\frac{1}{n}\sum [y \log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Multi-class Classification | Categorical Cross-Entropy | $-\frac{1}{n}\sum_i \sum_c y_{ic} \log \hat{y}_{ic}$ |
| Regression | Mean Squared Error (MSE) | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| Regression (robust) | Mean Absolute Error (MAE) | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ |

---

## 📚 References
- [Scikit-learn: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html) — Official API and model reference
- [Bishop, C.M. — Pattern Recognition and Machine Learning (2006)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — Theoretical foundation
- [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Andrew Ng's supervised learning derivations
- [Deep Learning Book — Goodfellow et al., Chapter 5](https://www.deeplearningbook.org/contents/ml.html) — Machine learning basics and generalization theory