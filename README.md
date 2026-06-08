# Multilevel Optimization in Machine Learning  
### A Comparative Study on Dense and Sparse Data

## Overview
This project studies computational methods for machine learning, focusing on how optimization techniques and multilevel strategies behave across different data structures.

Two datasets are used:
- MNIST (dense image data for classification)
- MovieLens (sparse user–item data for recommender systems)

The goal is not to compare datasets directly, but to evaluate how optimization methods, convergence behavior, and multilevel techniques perform in both environments.

---

## Objectives
- Analyze iterative optimization methods (Gradient Descent, SGD, Adam)
- Study convergence behavior and training stability
- Compare linear (least squares) and nonlinear (neural networks) models
- Evaluate multilevel training strategies (coarse → fine)
- Understand behavior on dense vs sparse data

---

## Project Structure
\usepackage{listings}

\begin{lstlisting}
computational-ml-project/
│
├── notebooks/
│   ├── mnist_analysis.ipynb
│   ├── movielens_analysis.ipynb
│   └── comparison_analysis.ipynb
│
├── src/
├── data/
├── results/
├── figures/
├── report/
├── requirements.txt
└── README.md
\end{lstlisting}


---

## Methodology

### Least Squares (Baseline)
Classification and prediction are formulated as:
\[
\min_W \|XW - Y\|^2
\]
Solved using iterative methods (gradient descent).

### Neural Networks
- MNIST → MLP / CNN
- MovieLens → Neural recommender (embeddings)

Optimizers:
- SGD
- Adam

### Multilevel Methods
- MNIST: low-resolution → high-resolution
- MovieLens: small embeddings → large embeddings

Goal: improve convergence and efficiency.

---

## Evaluation

### Task Metrics
- MNIST → Accuracy
- MovieLens → RMSE

### Computational Metrics
- Convergence rate (loss vs iterations)
- Number of iterations to convergence
- Training time
- Stability of optimization
- Multilevel speedup

---

## Important Note
This project does not compare datasets directly.  
It compares computational methods across:
- Dense data (MNIST)
- Sparse data (MovieLens)

---

## How to Run

1. Install dependencies:


2. Run notebooks in order:
- notebooks/mnist_analysis.ipynb
- notebooks/movielens_analysis.ipynb
- notebooks/comparison_analysis.ipynb

---

## Course Alignment
This project applies:
- Linear algebra (least squares)
- Iterative optimization methods
- Convergence analysis
- Multilevel methods
- Neural networks as nonlinear optimization

---

## Author
Edwin Mutimba  
Portland State University
