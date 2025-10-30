# Mathematical Theory: Optimal Transport for Dynamic Brain Connectivity

## 1. Introduction

This document provides the mathematical foundations for analyzing time-varying functional brain connectivity using optimal transport theory on the manifold of correlation/covariance matrices.

## 2. The Manifold of SPD Matrices

### 2.1 Definition

Functional connectivity matrices are **symmetric positive definite (SPD)** matrices:

```
S_++ ^n = {C ∈ R^{n×n} : C = C^T, v^T C v > 0 for all v ≠ 0}
```

where `n` is the number of brain regions.

### 2.2 Riemannian Structure

The set `S_++ ^n` forms a **Riemannian manifold** when equipped with the affine-invariant metric:

```
⟨V, W⟩_C = Tr(C^{-1} V C^{-1} W)
```

for tangent vectors `V, W ∈ T_C(S_++ ^n)` at point `C`.

**Key Properties:**
- **Geodesically complete**: Every pair of points connected by unique geodesic
- **Non-positive curvature**: Implies unique geodesics and convex distance functions
- **Affine-invariant**: Metric preserved under congruence transformations

## 3. Geodesics on the SPD Manifold

### 3.1 Geodesic Equation

The geodesic from `C_1` to `C_2` at parameter `t ∈ [0,1]` is:

```
γ(t) = C_1^{1/2} (C_1^{-1/2} C_2 C_1^{-1/2})^t C_1^{1/2}
```

**Equivalent formulation** (log-Euclidean):

```
γ(t) = exp((1-t) log(C_1) + t log(C_2))
```

where `exp` and `log` are matrix exponential and logarithm.

### 3.2 Geodesic Distance

The Riemannian distance between `C_1` and `C_2` is:

```
d(C_1, C_2) = ||log(C_1^{-1/2} C_2 C_1^{-1/2})||_F
            = sqrt(∑_i log^2(λ_i))
```

where `λ_i` are eigenvalues of `C_1^{-1} C_2`, and `||·||_F` is the Frobenius norm.

**Properties:**
- `d(C_1, C_2) ≥ 0` with equality iff `C_1 = C_2`
- `d(C_1, C_2) = d(C_2, C_1)` (symmetry)
- `d(C_1, C_3) ≤ d(C_1, C_2) + d(C_2, C_3)` (triangle inequality)
- `d(ACA^T, ADA^T) = d(C, D)` for invertible `A` (affine invariance)

## 4. Wasserstein Distance for SPD Matrices

### 4.1 Motivation

When viewing connectivity matrices as representing Gaussian distributions (zero-mean with covariance `C`), the **Wasserstein-2 distance** (W2) quantifies optimal transport cost between distributions.

### 4.2 Wasserstein-2 Distance

For zero-mean Gaussian distributions with covariances `C_1, C_2`:

```
W_2^2(N(0, C_1), N(0, C_2)) = Tr(C_1 + C_2 - 2(C_1^{1/2} C_2 C_1^{1/2})^{1/2})
```

This is the **Bures-Wasserstein metric**.

**Alternative (Frobenius on Square Roots):**

```
W_2(C_1, C_2) = ||C_1^{1/2} - C_2^{1/2}||_F
```

This provides a computational upper bound on the Riemannian distance and is often preferred for efficiency.

### 4.3 Relationship to Riemannian Distance

For SPD matrices:

```
W_2(C_1, C_2) ≥ d(C_1, C_2)
```

with equality when matrices commute (`C_1 C_2 = C_2 C_1`).

## 5. Wasserstein Barycenters

### 5.1 Definition

The **Wasserstein barycenter** (Fréchet mean) of SPD matrices `{C_1, ..., C_m}` with weights `{w_1, ..., w_m}` solves:

```
C* = argmin_C ∑_i w_i W_2^2(C, C_i)
```

### 5.2 Computation

**For Frobenius metric (closed form):**

```
C* = (∑_i w_i C_i^{1/2})^2
```

**For Riemannian metric (iterative):**

Use gradient descent on the manifold:

```
C_{k+1} = C_k^{1/2} exp(α ∑_i w_i log(C_k^{-1/2} C_i C_k^{-1/2})) C_k^{1/2}
```

where `α` is step size.

## 6. Dynamic Connectivity Framework

### 6.1 Sliding Window Estimation

Given fMRI time series `X(t) ∈ R^n`:

```
C(τ) = Cov(X(t) : t ∈ [τ - w/2, τ + w/2])
```

yields time-varying connectivity `{C(τ_1), ..., C(τ_T)}`.

### 6.2 Network Evolution Trajectory

Define the **Wasserstein trajectory**:

```
d(τ) = W_2(C(τ), C(τ + Δτ))
```

**Interpretation:**
- Low `d(τ)`: Network stable at time `τ`
- High `d(τ)`: Network undergoing reconfiguration
- Peaks indicate state transitions

### 6.3 State Segmentation

Cluster connectivity matrices `{C(τ)}` into discrete states using:

**K-means on SPD manifold:**
1. Initialize cluster centers `{μ_1, ..., μ_k}`
2. Assign: `s(τ) = argmin_k W_2(C(τ), μ_k)`
3. Update: `μ_k = Barycenter({C(τ) : s(τ) = k})`
4. Repeat until convergence

## 7. Theoretical Guarantees

### 7.1 Convergence of Barycenters

**Theorem**: The Wasserstein barycenter algorithm converges to a unique minimum when using the affine-invariant metric on `S_++ ^n`.

**Proof sketch**: The objective function is geodesically convex on the non-positively curved manifold, guaranteeing unique local minimum = global minimum.

### 7.2 Stability Under Noise

**Proposition**: For noisy estimates `C̃ = C + E` where `||E||_F ≤ ε`:

```
W_2(C, C̃) ≤ K ε
```

for constant `K` depending on condition number of `C`.

This ensures Wasserstein distances are robust to estimation noise.

### 7.3 Geodesic Interpolation Properties

**Lemma**: Geodesic interpolation preserves positive definiteness:

```
If C_1, C_2 ∈ S_++ ^n, then γ(t) ∈ S_++ ^n for all t ∈ [0,1]
```

**Proof**: Using spectral decomposition, eigenvalues interpolate geometrically (remaining positive).

## 8. Computational Complexity

### Operations Complexity Table

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Matrix sqrt | O(n³) | Via eigendecomposition |
| Geodesic point | O(n³) | Two matrix sqrts + mult |
| Wasserstein distance | O(n³) | One geodesic point |
| Barycenter (Frobenius) | O(mn³) | Closed form with m matrices |
| Barycenter (Riemannian) | O(kmn³) | k iterations |
| K-means segmentation | O(TKmn³) | T windows, K states, m iterations |

### Numerical Stability

**Condition number considerations:**
- Eigenvalue computations stable for well-conditioned SPD matrices
- Use log-domain for extreme eigenvalue ratios
- Regularization (shrinkage) recommended for short time windows

## 9. Extensions and Future Directions

### 9.1 Entropic Regularization (Sinkhorn)

Replace exact optimal transport with regularized version:

```
W_2,ε(C_1, C_2) = min_{π} ⟨π, M⟩ + ε H(π)
```

where `H(π)` is entropy. Enables GPU acceleration.

### 9.2 Transport Plans

The optimal transport plan `π*` reveals which brain regions in state 1 correspond to regions in state 2, providing interpretable network reorganization patterns.

### 9.3 Higher-Order Moments

Extend beyond covariance to:
- Coskewness tensors (3rd order)
- Cokurtosis tensors (4th order)

Using tensor-valued optimal transport.

## 10. References

1. **Bhatia, R.** (2009). *Positive Definite Matrices*. Princeton University Press.
2. **Peyré, G., & Cuturi, M.** (2019). Computational optimal transport. *Foundations and Trends in Machine Learning*.
3. **Pennec, X., Fillard, P., & Ayache, N.** (2006). A Riemannian framework for tensor computing. *International Journal of Computer Vision*.
4. **Agueh, M., & Carlier, G.** (2011). Barycenters in the Wasserstein space. *SIAM Journal on Mathematical Analysis*.
5. **Allen, E. A., et al.** (2014). Tracking whole-brain connectivity dynamics in the resting state. *Cerebral Cortex*.

## Appendix: Notation Summary

| Symbol | Meaning |
|--------|---------|
| `S_++ ^n` | Manifold of n×n SPD matrices |
| `C, Σ` | Covariance/correlation matrix |
| `W_2` | Wasserstein-2 distance |
| `d` | Riemannian distance |
| `γ(t)` | Geodesic path |
| `exp, log` | Matrix exponential/logarithm |
| `||·||_F` | Frobenius norm |
| `Tr` | Matrix trace |
| `⊗` | Tensor/outer product |
