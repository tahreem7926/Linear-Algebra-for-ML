"""
Chapter 7 -- Singular Value Decomposition (SVD)
Companion code to notes/07_singular_value_decomposition.md

Covers:
  - SVD constructed from scratch via eigendecomposition of A^T A (Section 7.2)
  - Cross-check against np.linalg.svd
  - Low-rank approximation / Eckart-Young (Section 7.5) via image compression
  - PCA via SVD directly on the data matrix, vs. via eigendecomposition of the
    covariance matrix (Section 7.7) -- confirms they agree, and that SVD avoids
    ever forming X^T X
  - Moore-Penrose pseudoinverse for a rank-deficient least-squares problem (Section 7.6)
"""
import numpy as np


def svd_from_scratch(A: np.ndarray):
    """Section 7.2's derivation, implemented directly: eigendecompose A^T A, derive
    singular values and V from it, then build U column-by-column as A v_i / sigma_i."""
    A = A.astype(float)
    m, n = A.shape

    AtA = A.T @ A
    eigvals, V = np.linalg.eigh(AtA)          # ascending order, V orthonormal (spectral theorem)
    order = np.argsort(eigvals)[::-1]          # descending: sigma_1 >= sigma_2 >= ...
    eigvals, V = eigvals[order], V[:, order]
    eigvals = np.clip(eigvals, 0, None)         # guard against tiny negative numerical noise
    singular_values = np.sqrt(eigvals)

    r = np.sum(singular_values > 1e-10)         # numerical rank
    U = np.zeros((m, m))
    for i in range(r):
        U[:, i] = (A @ V[:, i]) / singular_values[i]
    # Extend U to a full orthonormal basis of R^m if r < m (Section 7.2)
    if r < m:
        Q_extra, _ = np.linalg.qr(np.hstack([U[:, :r], np.eye(m)]))
        U[:, r:] = Q_extra[:, r:m]

    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, singular_values)
    return U, Sigma, V.T


def demo_worked_example():
    """Section 7.8."""
    A = np.array([[3.0, 0.0], [0.0, -2.0], [0.0, 0.0]])
    U, Sigma, Vt = svd_from_scratch(A)
    print("Singular values:", np.diag(Sigma)[:2])
    assert np.allclose(sorted(np.diag(Sigma)[:2], reverse=True), [3.0, 2.0])
    reconstructed = U @ Sigma @ Vt
    print("Reconstructed A =\n", np.round(reconstructed, 8))
    assert np.allclose(reconstructed, A, atol=1e-8)
    print("From-scratch SVD reconstructs A exactly. Confirmed.\n")


def demo_cross_check_random_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 4))
    U, Sigma, Vt = svd_from_scratch(A)
    U_np, s_np, Vt_np = np.linalg.svd(A)

    print("From-scratch singular values:", np.round(np.diag(Sigma)[:4], 6))
    print("np.linalg.svd singular values:", np.round(s_np, 6))
    assert np.allclose(sorted(np.diag(Sigma)[:4]), sorted(s_np), atol=1e-6)
    assert np.allclose(U @ Sigma @ Vt, A, atol=1e-8)
    print("Singular values agree with np.linalg.svd; reconstruction confirmed.\n")


def demo_low_rank_approximation():
    """Section 7.5: truncated SVD is the provably-best rank-k approximation."""
    rng = np.random.default_rng(1)
    # Build a matrix with fast-decaying singular values (simulates a "compressible" signal)
    m, n = 40, 30
    U0, _ = np.linalg.qr(rng.normal(size=(m, m)))
    V0, _ = np.linalg.qr(rng.normal(size=(n, n)))
    true_singular_values = np.array([50, 20, 8, 3, 1] + [0.05] * (min(m, n) - 5))
    Sigma0 = np.zeros((m, n))
    np.fill_diagonal(Sigma0, true_singular_values)
    A = U0 @ Sigma0 @ V0.T

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    for k in [1, 3, 5, 10]:
        A_k = (U[:, :k] * s[:k]) @ Vt[:k, :]
        frob_error = np.linalg.norm(A - A_k, ord='fro')
        # Eckart-Young: the error of the best rank-k approximation equals sqrt(sum of squared
        # DROPPED singular values)
        predicted_error = np.sqrt(np.sum(s[k:] ** 2))
        print(f"k={k:2d}: ||A - A_k||_F = {frob_error:.4f}  "
              f"(predicted from dropped singular values: {predicted_error:.4f})")
        assert np.isclose(frob_error, predicted_error, atol=1e-8)
    print("Eckart-Young error formula confirmed for every truncation level k.\n")


def demo_pca_via_svd_vs_eigendecomposition():
    """
    Section 7.7: PCA via SVD of the data matrix directly agrees with PCA via
    eigendecomposition of the covariance matrix -- but SVD never forms X^T X.
    """
    rng = np.random.default_rng(2)
    n_samples = 300
    z = rng.normal(size=n_samples)
    X = np.column_stack([
        z + 0.05 * rng.normal(size=n_samples),
        0.6 * z + 0.05 * rng.normal(size=n_samples),
        0.05 * rng.normal(size=n_samples),
    ])
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Route 1: eigendecomposition of the covariance matrix (Chapter 5's approach)
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Route 2: SVD directly on X_centered (never forms X^T X)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    singular_value_variances = (s ** 2) / (n_samples - 1)

    print("Eigenvalues from covariance eigendecomposition:", np.round(eigvals, 5))
    print("sigma_i^2/(n-1) from SVD of X directly:         ", np.round(singular_value_variances, 5))
    assert np.allclose(eigvals, singular_value_variances, atol=1e-6)

    # Principal component directions should agree up to sign
    for i in range(3):
        agreement = abs(eigvecs[:, i] @ Vt[i])
        assert np.isclose(agreement, 1.0, atol=1e-6)
    print("Both routes agree (component directions match up to sign). Confirmed.\n")


def demo_pseudoinverse_rank_deficient():
    """Section 7.6: pseudoinverse handles a rank-deficient least-squares problem
    that a plain QR-based solve (Chapter 6) can't handle cleanly (R becomes singular)."""
    # Columns 2 and 3 are identical -> A is rank-deficient (rank 2, not 3)
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    b = np.array([1.0, 2.0, 3.0, 0.0])

    print("rank(A) =", np.linalg.matrix_rank(A), "< n =", A.shape[1], "-- A is rank-deficient")
    x_pinv = np.linalg.pinv(A) @ b
    residual = np.linalg.norm(A @ x_pinv - b)
    print("Minimum-norm least-squares solution:", np.round(x_pinv, 4))
    print("Residual ||Ax - b||:", round(residual, 6))

    # Confirm it's the minimum-norm solution among all least-squares minimizers by comparing
    # against a solution nudged within the null space of A (should have equal or larger residual
    # and strictly larger norm, since x_pinv is the minimum-norm choice)
    _, _, Vt = np.linalg.svd(A)
    null_direction = Vt[-1]  # a vector in null(A), since A is rank 2 with 3 columns
    x_nudged = x_pinv + 0.5 * null_direction
    assert np.isclose(np.linalg.norm(A @ x_nudged - b), residual, atol=1e-8), "Same residual (still a minimizer)"
    assert np.linalg.norm(x_nudged) > np.linalg.norm(x_pinv), "But larger norm than the pseudoinverse solution"
    print("Pseudoinverse solution confirmed as the minimum-norm least-squares minimizer.\n")


if __name__ == "__main__":
    demo_worked_example()
    demo_cross_check_random_matrix()
    demo_low_rank_approximation()
    demo_pca_via_svd_vs_eigendecomposition()
    demo_pseudoinverse_rank_deficient()
    print("All checks passed.")
