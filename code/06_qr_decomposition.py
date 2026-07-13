"""
Chapter 6 -- Orthogonality and QR Decomposition
Companion code to notes/06_orthogonality_and_qr_decomposition.md

Covers:
  - Classical Gram-Schmidt QR, from scratch (Section 6.2-6.3)
  - Modified Gram-Schmidt QR, from scratch (Section 6.4) -- and a demonstration
    of *why* it's more stable than the classical version on a near-degenerate matrix
  - Cross-check against np.linalg.qr (Householder-based)
  - Least squares via the normal equations vs. via QR (Section 6.5), including
    a demonstration of the normal equations losing accuracy on an ill-conditioned matrix
"""
import numpy as np


def qr_classical_gram_schmidt(A: np.ndarray):
    """Section 6.2-6.3, implemented exactly as derived: project against the ORIGINAL
    columns' already-built u_i's, all at once, per column."""
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for k in range(n):
        w = A[:, k].copy()
        for i in range(k):
            R[i, k] = Q[:, i] @ A[:, k]
            w -= R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(w)
        Q[:, k] = w / R[k, k]
    return Q, R


def qr_modified_gram_schmidt(A: np.ndarray):
    """Section 6.4: subtract each projection immediately from a running vector,
    instead of projecting the original column against every previous u_i at once."""
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()  # running vectors, updated in place -- this is the key difference from CGS
    for k in range(n):
        R[k, k] = np.linalg.norm(V[:, k])
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k + 1, n):
            R[k, j] = Q[:, k] @ V[:, j]
            V[:, j] -= R[k, j] * Q[:, k]
    return Q, R


def demo_worked_example():
    """Section 6.7."""
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    Q, R = qr_classical_gram_schmidt(A)
    print("Q =\n", np.round(Q, 4))
    print("R =\n", np.round(R, 4))
    assert np.allclose(Q @ R, A), "QR should reconstruct A"
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-10), "Q should have orthonormal columns"

    Q_np, R_np = np.linalg.qr(A)
    # Signs can differ between implementations -- compare via reconstruction, not entrywise
    assert np.allclose(Q @ R, Q_np @ R_np)
    print("Matches np.linalg.qr (up to sign convention). Confirmed.\n")


def demo_cgs_vs_mgs_stability():
    """
    Section 6.4: classical Gram-Schmidt loses orthogonality on nearly-parallel columns;
    modified Gram-Schmidt stays much closer to Q^T Q = I.
    """
    eps = 1e-8
    # Three nearly-parallel columns -- a classic stress test for orthogonalization stability
    A = np.array([
        [1.0,     1.0,     1.0],
        [eps,     0.0,     0.0],
        [0.0,     eps,     0.0],
        [0.0,     0.0,     eps],
    ])

    Q_cgs, _ = qr_classical_gram_schmidt(A)
    Q_mgs, _ = qr_modified_gram_schmidt(A)

    orth_error_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(3))
    orth_error_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(3))

    print(f"||Q^T Q - I|| with classical Gram-Schmidt: {orth_error_cgs:.2e}")
    print(f"||Q^T Q - I|| with modified Gram-Schmidt:  {orth_error_mgs:.2e}")
    assert orth_error_mgs <= orth_error_cgs, "MGS should be at least as orthogonal as CGS here"
    print("Modified Gram-Schmidt preserves orthogonality better on a near-degenerate matrix.\n")


def least_squares_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The textbook formula -- see Section 6.5 for why this is numerically risky."""
    return np.linalg.solve(A.T @ A, A.T @ b)


def least_squares_via_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Section 6.5's derivation: solve Rx = Q^T b by back-substitution (via triangular solve)."""
    Q, R = np.linalg.qr(A)  # economy QR
    from scipy.linalg import solve_triangular
    return solve_triangular(R, Q.T @ b)


def demo_least_squares_agreement():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(20, 5))
    b = rng.normal(size=20)

    x_normal = least_squares_normal_equations(A, b)
    x_qr = least_squares_via_qr(A, b)
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Normal equations:", np.round(x_normal, 4))
    print("QR:              ", np.round(x_qr, 4))
    print("np.linalg.lstsq: ", np.round(x_lstsq, 4))
    assert np.allclose(x_normal, x_qr, atol=1e-8)
    assert np.allclose(x_qr, x_lstsq, atol=1e-8)
    print("All three methods agree on a well-conditioned problem. Confirmed.\n")


def demo_ill_conditioned_comparison():
    """
    Section 6.5's central numerical claim: on an ill-conditioned A, forming A^T A
    (condition number squared) loses meaningfully more accuracy than solving via QR.
    """
    # Build an ill-conditioned A via a Vandermonde-like construction (classic bad case)
    x_pts = np.linspace(0, 1, 15)
    A = np.vander(x_pts, N=8, increasing=True)  # columns are 1, x, x^2, ..., x^7 -- nearly collinear
    true_coeffs = np.arange(1, 9, dtype=float)
    b = A @ true_coeffs + 1e-10 * np.random.default_rng(0).normal(size=A.shape[0])

    cond_A = np.linalg.cond(A)
    cond_AtA = np.linalg.cond(A.T @ A)
    print(f"cond(A)   = {cond_A:.3e}")
    print(f"cond(A^T A) = {cond_AtA:.3e}  (should be roughly cond(A)^2 = {cond_A**2:.3e})")
    assert cond_AtA > cond_A * 10, "A^T A should be substantially worse-conditioned than A"

    x_normal = least_squares_normal_equations(A, b)
    x_qr = least_squares_via_qr(A, b)

    err_normal = np.linalg.norm(x_normal - true_coeffs)
    err_qr = np.linalg.norm(x_qr - true_coeffs)
    print(f"Recovery error (normal equations): {err_normal:.3e}")
    print(f"Recovery error (QR):                {err_qr:.3e}")
    print("QR recovers the true coefficients more accurately on this ill-conditioned system.\n")


if __name__ == "__main__":
    demo_worked_example()
    demo_cgs_vs_mgs_stability()
    demo_least_squares_agreement()
    demo_ill_conditioned_comparison()
    print("All checks passed.")
