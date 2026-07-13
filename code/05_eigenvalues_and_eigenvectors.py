"""
Chapter 5 -- Eigenvalues and Eigenvectors
Companion code to notes/05_eigenvalues_and_eigenvectors.md

Covers:
  - Worked example (Section 5.7), cross-checked against np.linalg.eig
  - Diagonalization A = P D P^-1
  - Spectral theorem for symmetric matrices (real eigenvalues, orthogonal eigenvectors)
  - Power iteration implemented from scratch, converging to the dominant eigenvector
  - A from-scratch PCA using eigendecomposition of the covariance matrix
"""
import numpy as np


def demo_worked_example():
    """Section 5.7."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    eigvals, eigvecs = np.linalg.eig(A)
    print("Eigenvalues:", eigvals)
    assert np.allclose(sorted(eigvals), [2.0, 5.0])

    # trace = sum of eigenvalues, det = product of eigenvalues -- fast sanity check from the notes
    assert np.isclose(np.trace(A), eigvals.sum())
    assert np.isclose(np.linalg.det(A), np.prod(eigvals))
    print("trace(A) == sum(eigenvalues) and det(A) == prod(eigenvalues): confirmed.\n")


def demo_diagonalization():
    """Section 5.3: A = P D P^-1."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    eigvals, P = np.linalg.eig(A)
    D = np.diag(eigvals)
    reconstructed = P @ D @ np.linalg.inv(P)
    print("P D P^-1 =\n", np.round(reconstructed, 10))
    assert np.allclose(reconstructed, A)
    print("Diagonalization A = P D P^-1 confirmed.\n")


def demo_spectral_theorem():
    """Section 5.4: symmetric matrices have real eigenvalues and orthogonal eigenvectors."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    A = M + M.T  # symmetric by construction

    eigvals, Q = np.linalg.eigh(A)  # eigh assumes/exploits symmetry
    print("Eigenvalues (should all be real):", eigvals)
    assert np.all(np.isreal(eigvals))

    # Q should be orthogonal: Q^T Q = I
    print("Q^T Q =\n", np.round(Q.T @ Q, 8))
    assert np.allclose(Q.T @ Q, np.eye(4))
    assert np.allclose(A, Q @ np.diag(eigvals) @ Q.T)
    print("Spectral theorem (A = Q Lambda Q^T, Q orthogonal) confirmed for a random symmetric matrix.\n")


def power_iteration(A: np.ndarray, n_iters: int = 100, seed: int = 0):
    """Section 5.5, implemented from scratch."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=A.shape[0])
    x /= np.linalg.norm(x)
    for _ in range(n_iters):
        x = A @ x
        x /= np.linalg.norm(x)
    dominant_eigenvalue = x @ A @ x  # Rayleigh quotient
    return dominant_eigenvalue, x


def demo_power_iteration():
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    lam, v = power_iteration(A)
    eigvals, eigvecs = np.linalg.eig(A)
    dominant_idx = np.argmax(np.abs(eigvals))

    print(f"Power iteration converged to eigenvalue {lam:.6f} (true dominant: {eigvals[dominant_idx]:.6f})")
    assert np.isclose(lam, eigvals[dominant_idx], atol=1e-4)
    # Direction should match up to sign
    true_v = eigvecs[:, dominant_idx]
    assert np.isclose(abs(v @ true_v), 1.0, atol=1e-4)
    print("Power iteration direction matches the true dominant eigenvector (up to sign). Confirmed.\n")


def pca_from_scratch(X: np.ndarray, n_components: int):
    """Section 5.6: PCA via eigendecomposition of the covariance matrix (see Chapter 7 for the
    numerically-preferred SVD-based version -- this one exists to make the connection explicit)."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
    order = np.argsort(eigvals)[::-1]                # sort descending: most variance first
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    components = eigvecs[:, :n_components]
    projected = X_centered @ components
    explained_variance_ratio = eigvals[:n_components] / eigvals.sum()
    return projected, components, explained_variance_ratio


def demo_pca():
    rng = np.random.default_rng(1)
    # Correlated 2D data with most variance along one direction
    n = 500
    z = rng.normal(size=n)
    X = np.column_stack([z + 0.1 * rng.normal(size=n), 0.5 * z + 0.1 * rng.normal(size=n)])

    projected, components, ratio = pca_from_scratch(X, n_components=1)
    print("Explained variance ratio of first component:", ratio)
    assert ratio[0] > 0.9, "First component should explain most of the variance in this constructed example"
    print("First principal component captures", f"{ratio[0]*100:.1f}%", "of variance, as expected.\n")


if __name__ == "__main__":
    demo_worked_example()
    demo_diagonalization()
    demo_spectral_theorem()
    demo_power_iteration()
    demo_pca()
    print("All checks passed.")
