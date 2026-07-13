"""
Chapter 4 -- Basis and Change of Basis
Companion code to notes/04_basis_and_change_of_basis.md

Covers:
  - The change-of-basis matrix P and its use (worked example from the notes)
  - Similarity transform A' = P^-1 A P for a linear map
  - Orthonormal basis change: P^-1 == P^T
  - Revisits the shadow-projection matrix from the intro exercise to contrast
    "change of basis" (invertible) against "projection" (rank-deficient, not invertible)
"""
import numpy as np


def demo_change_of_basis_worked_example():
    """Section 4.7."""
    P = np.array([[1.0, -1.0],
                  [1.0, 1.0]])
    P_inv = np.linalg.inv(P)

    x_old = np.array([3.0, 1.0])
    x_new = P_inv @ x_old
    print("x_old =", x_old, " -> x_new (new-basis coordinates) =", x_new)
    assert np.allclose(x_new, [2.0, -1.0]), "Should match the hand-derived (2, -1) from the notes"

    # Reconstruct: 2*b1 - 1*b2 should give back x_old
    b1, b2 = P[:, 0], P[:, 1]
    reconstructed = x_new[0] * b1 + x_new[1] * b2
    assert np.allclose(reconstructed, x_old)
    print("Reconstruction from new-basis coordinates matches x_old. Confirmed.\n")


def demo_similarity_transform():
    """Section 4.3: A' = P^-1 A P represents the same linear map in a new basis."""
    A = np.array([[2.0, 1.0],
                  [0.0, 3.0]])
    P = np.array([[1.0, -1.0],
                  [1.0, 1.0]])
    P_inv = np.linalg.inv(P)
    A_prime = P_inv @ A @ P

    # Similar matrices share eigenvalues, trace, and determinant -- properties of the MAP, not the coordinates
    eig_A = np.sort(np.linalg.eigvals(A))
    eig_Aprime = np.sort(np.linalg.eigvals(A_prime))
    print("Eigenvalues of A: ", eig_A)
    print("Eigenvalues of A':", eig_Aprime)
    assert np.allclose(eig_A, eig_Aprime)
    assert np.isclose(np.trace(A), np.trace(A_prime))
    assert np.isclose(np.linalg.det(A), np.linalg.det(A_prime))
    print("Eigenvalues, trace, and determinant preserved under similarity transform. Confirmed.\n")


def demo_orthonormal_basis_change():
    """Section 4.4: for orthonormal P, P^-1 == P^T (no matrix inversion needed)."""
    theta = np.pi / 6
    P = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # a rotation matrix: orthonormal columns by construction

    print("P^T P =\n", np.round(P.T @ P, 10))
    assert np.allclose(P.T @ P, np.eye(2))
    assert np.allclose(np.linalg.inv(P), P.T)
    print("P^-1 == P^T confirmed for this orthogonal (rotation) matrix.\n")


def demo_change_of_basis_vs_projection():
    """
    Section 4.5 -- contrasts an invertible change-of-basis matrix against the
    (rank-deficient, non-invertible) shadow-projection matrix from the quiz this
    chapter's notes reference directly.
    """
    # Change-of-basis matrix: square, invertible
    P = np.array([[1.0, -1.0], [1.0, 1.0]])
    print("Change-of-basis P is square:", P.shape, " invertible:", not np.isclose(np.linalg.det(P), 0))

    # Shadow-projection matrix A (2x3, rank-deficient) for s_hat = (4/13, -3/13, -12/13)
    s = np.array([4 / 13, -3 / 13, -12 / 13])
    A = np.array([
        [1.0, 0.0, -s[0] / s[2]],
        [0.0, 1.0, -s[1] / s[2]],
    ])
    print("Shadow-projection A is 2x3 (non-square), so 'inverting' it doesn't even typecheck.")
    r = np.array([6.0, 2.0, 3.0])
    r_prime = A @ r
    print("A @ r =", r_prime, "(matches r' = (7, 5/4) from the shadow-projection formula)")
    assert np.allclose(r_prime, [7.0, 5.0 / 4.0])

    # The point: A discards information. Two different 3D points can share the same shadow.
    r2 = r + 5 * s  # move 5 units further along the sun's ray -- same shadow, different point
    r2_prime = A @ r2
    print("A @ (r + 5*s_hat) =", r2_prime, "-- same shadow, confirming A is NOT invertible / information-preserving.\n")
    assert np.allclose(r_prime, r2_prime)


if __name__ == "__main__":
    demo_change_of_basis_worked_example()
    demo_similarity_transform()
    demo_orthonormal_basis_change()
    demo_change_of_basis_vs_projection()
    print("All checks passed.")
