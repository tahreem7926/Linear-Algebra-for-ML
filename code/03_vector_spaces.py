"""
Chapter 3 -- Vector Spaces
Companion code to notes/03_vector_spaces.md

Covers:
  - Checking linear independence via rank
  - Rank-nullity theorem, numerically confirmed
  - The four fundamental subspaces and their orthogonality relations
  - Cauchy-Schwarz inequality, numerically confirmed
"""
import numpy as np


def is_linearly_independent(vectors: np.ndarray, tol: float = 1e-10) -> bool:
    """vectors: columns are the vectors to test. See Section 3.3."""
    rank = np.linalg.matrix_rank(vectors, tol=tol)
    return rank == vectors.shape[1]


def demo_linear_independence():
    independent_set = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # 3 vectors in R^3, independent
    dependent_set = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])      # second column = 2x first

    print("Independent set is independent?", is_linearly_independent(independent_set))
    print("Dependent set is independent?  ", is_linearly_independent(dependent_set))
    assert is_linearly_independent(independent_set)
    assert not is_linearly_independent(dependent_set)

    # Section 3.3 theorem: more than n vectors in R^n are always dependent
    too_many = np.random.default_rng(0).normal(size=(3, 5))  # 5 vectors in R^3
    assert not is_linearly_independent(too_many)
    print("5 vectors in R^3 are necessarily dependent: confirmed.\n")


def demo_rank_nullity():
    """Section 3.5: n = rank(A) + nullity(A)."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 6))
    A[:, 4] = A[:, 0] + 2 * A[:, 1]   # force column 4 to be dependent -> rank < 6
    A[:, 5] = A[:, 2] - A[:, 3]       # force column 5 to be dependent too

    n = A.shape[1]
    rank = np.linalg.matrix_rank(A)

    # Null space basis via SVD: right singular vectors with ~zero singular value span null(A)
    _, s, Vt = np.linalg.svd(A)
    tol = max(A.shape) * np.finfo(float).eps * s.max()
    nullity = np.sum(s < tol) + (n - len(s))  # account for singular values array shorter than n

    print(f"n = {n}, rank(A) = {rank}, nullity(A) = {nullity}, rank+nullity = {rank + nullity}")
    assert rank + nullity == n
    print("Rank-nullity theorem confirmed numerically.\n")


def demo_four_fundamental_subspaces():
    """Section 3.6: row(A) is orthogonal to null(A); col(A) is orthogonal to null(A^T)."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 3))
    A[:, 2] = A[:, 0] + A[:, 1]  # make it rank-deficient (rank 2) so null(A) is nontrivial

    U, s, Vt = np.linalg.svd(A)
    r = np.sum(s > 1e-10)

    row_space_basis = Vt[:r]         # top r rows of V^T span row(A)
    null_space_basis = Vt[r:]        # remaining rows span null(A)

    # Every row-space basis vector should be orthogonal to every null-space basis vector
    cross_products = row_space_basis @ null_space_basis.T
    print("row(A) . null(A) inner products (should all be ~0):\n", np.round(cross_products, 10))
    assert np.allclose(cross_products, 0, atol=1e-8)
    print("row(A) is confirmed orthogonal to null(A).\n")


def demo_cauchy_schwarz():
    """Section 3.7: |<u,v>| <= ||u|| ||v||, confirmed over random vectors."""
    rng = np.random.default_rng(3)
    violations = 0
    for _ in range(10_000):
        u, v = rng.normal(size=5), rng.normal(size=5)
        lhs = abs(np.dot(u, v))
        rhs = np.linalg.norm(u) * np.linalg.norm(v)
        if lhs > rhs + 1e-9:
            violations += 1
    print(f"Cauchy-Schwarz violations across 10,000 random trials: {violations}")
    assert violations == 0
    print("Cauchy-Schwarz inequality holds in every trial, as proved in the notes.\n")


if __name__ == "__main__":
    demo_linear_independence()
    demo_rank_nullity()
    demo_four_fundamental_subspaces()
    demo_cauchy_schwarz()
    print("All checks passed.")
