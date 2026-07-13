"""
Chapter 2 -- Matrices and Operations
Companion code to notes/02_matrices_and_operations.md

Covers:
  - Matrix multiplication as composition (non-commutativity, worked example)
  - Determinant as volume scaling (numerically demonstrated)
  - Trace cyclic property
  - Positive semi-definiteness of a covariance/Gram matrix
  - einsum as the general-purpose tool for index-notation expressions
"""
import numpy as np


def demo_composition_noncommutativity():
    """Section 2.9: AB (stretch then rotate) != BA (rotate then stretch)."""
    A = np.array([[0.0, -1.0], [1.0, 0.0]])   # 90-degree rotation
    B = np.array([[2.0, 0.0], [0.0, 1.0]])    # stretch x by 2

    AB = A @ B
    BA = B @ A
    print("AB (stretch then rotate) =\n", AB)
    print("BA (rotate then stretch) =\n", BA)
    assert not np.allclose(AB, BA), "Expected AB != BA -- composition order matters!"

    # Confirm the composition interpretation directly on a test vector
    x = np.array([1.0, 0.0])
    assert np.allclose(A @ (B @ x), (A @ B) @ x), "(AB)x should equal A(Bx)"
    print("(AB)x == A(Bx): composition <-> multiplication confirmed.\n")


def demo_determinant_as_volume():
    """Section 2.5: |det(A)| = volume scaling factor of the unit square under A."""
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    det = np.linalg.det(A)

    # Monte Carlo estimate of area scaling: sample points in unit square, transform, compare areas
    rng = np.random.default_rng(0)
    n_samples = 200_000
    pts = rng.uniform(0, 1, size=(n_samples, 2))
    transformed = pts @ A.T
    # Area of unit square = 1; area of transformed parallelogram approximated via convex hull
    from scipy.spatial import ConvexHull
    hull_area = ConvexHull(transformed).volume  # 'volume' is the 2D area for a 2D hull
    print(f"det(A) = {det:.4f}  |det(A)| = {abs(det):.4f}")
    print(f"Transformed unit square area (convex hull) = {hull_area:.4f}")
    assert np.isclose(abs(det), hull_area, atol=1e-2)
    print("Confirms |det(A)| == area scaling factor.\n")


def demo_trace_cyclic_property():
    """Section 2.6: tr(AB) == tr(BA) even though AB != BA."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 3))
    # Note: AB is 3x3 and BA is 4x4 -- not even the same shape, let alone equal -- yet their traces agree.
    trAB = np.trace(A @ B)
    trBA = np.trace(B @ A)
    print(f"tr(AB) = {trAB:.6f}, tr(BA) = {trBA:.6f}")
    assert np.isclose(trAB, trBA)
    print("Cyclic trace property confirmed.\n")


def demo_psd_covariance():
    """Section 2.7 / 2.8: covariance matrices are always positive semi-definite."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 5))  # 500 samples, 5 features
    X_centered = X - X.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    eigvals = np.linalg.eigvalsh(cov)  # eigvalsh: symmetric-matrix-specific, faster & more stable
    print("Covariance matrix eigenvalues:", np.round(eigvals, 4))
    assert np.all(eigvals >= -1e-10), "Covariance matrix should be PSD (all eigenvalues >= 0)!"
    print("All eigenvalues >= 0: covariance matrix confirmed PSD.\n")


def demo_einsum_for_index_notation():
    """einsum is the practical tool for the Einstein-summation expressions used throughout the notes."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 5))

    # (AB)_ik = sum_j A_ij B_jk  -- exactly Section 2.2's definition, written directly in einsum
    manual_matmul = np.einsum('ij,jk->ik', A, B)
    assert np.allclose(manual_matmul, A @ B)
    print("einsum('ij,jk->ik', A, B) matches A @ B: index-notation definition confirmed.")


if __name__ == "__main__":
    demo_composition_noncommutativity()
    demo_determinant_as_volume()
    demo_trace_cyclic_property()
    demo_psd_covariance()
    demo_einsum_for_index_notation()
    print("\nAll checks passed.")
