"""
Chapter 1 -- Systems of Linear Equations
Companion code to notes/01_systems_of_linear_equations.md

Covers:
  - Gaussian elimination from scratch (row echelon form + back-substitution)
  - Cross-check against np.linalg.solve
  - Rank / consistency check (Rouche-Capelli)
  - LU decomposition and reuse across multiple right-hand sides
"""
import numpy as np


def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b via Gaussian elimination with partial pivoting + back-substitution.
    Implemented from scratch to make every elementary row operation explicit --
    see notes/01_systems_of_linear_equations.md Section 1.2.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    aug = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination (with partial pivoting for numerical stability)
    for col in range(n):
        pivot_row = np.argmax(np.abs(aug[col:, col])) + col
        if np.isclose(aug[pivot_row, col], 0.0):
            raise ValueError("Matrix is singular (or nearly so) -- no unique solution.")
        aug[[col, pivot_row]] = aug[[pivot_row, col]]  # elementary op 1: row swap

        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            aug[row] -= factor * aug[col]              # elementary op 3: row - multiple of another row

    # Back-substitution
    x = np.zeros(n)
    for row in range(n - 1, -1, -1):
        x[row] = (aug[row, -1] - aug[row, row + 1:n] @ x[row + 1:n]) / aug[row, row]
    return x


def rank_consistency_check(A: np.ndarray, b: np.ndarray) -> bool:
    """Rouche-Capelli: consistent iff rank(A) == rank([A|b]). See Section 1.3."""
    augmented = np.hstack([A, b.reshape(-1, 1)])
    return np.linalg.matrix_rank(A) == np.linalg.matrix_rank(augmented)


def solve_via_lu_reuse(A: np.ndarray, b_list: list[np.ndarray]) -> list[np.ndarray]:
    """Factor once, solve for many right-hand sides cheaply. See Section 1.4."""
    from scipy.linalg import lu_factor, lu_solve
    lu, piv = lu_factor(A)
    return [lu_solve((lu, piv), b) for b in b_list]


if __name__ == "__main__":
    # Worked example from notes Section 1.6
    A = np.array([[2.0, 1.0],
                  [4.0, -1.0]])
    b = np.array([5.0, 1.0])

    x_manual = gaussian_elimination(A, b)
    x_numpy = np.linalg.solve(A, b)

    print("Manual Gaussian elimination:", x_manual)
    print("np.linalg.solve:            ", x_numpy)
    assert np.allclose(x_manual, x_numpy), "Manual solve disagrees with NumPy!"
    assert np.allclose(x_manual, [1.0, 3.0]), "Doesn't match the hand-derived answer in the notes!"
    print("Matches hand-derived answer (1, 3). Rank/consistency:", rank_consistency_check(A, b))

    # Underdetermined example: infinitely many solutions (rank(A) = rank([A|b]) < n)
    A2 = np.array([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0]])
    b2 = np.array([6.0, 12.0])
    print("\nUnderdetermined system consistent?", rank_consistency_check(A2, b2),
          "| rank(A2) =", np.linalg.matrix_rank(A2), "< n = 3 -> infinitely many solutions")

    # Inconsistent example: rank(A) < rank([A|b])
    A3 = np.array([[1.0, 2.0],
                   [2.0, 4.0]])
    b3 = np.array([1.0, 3.0])  # not a multiple of b3[0] matching the row scaling -> inconsistent
    print("Inconsistent system consistent?", rank_consistency_check(A3, b3),
          "(rank(A3) =", np.linalg.matrix_rank(A3),
          ", rank([A3|b3]) =", np.linalg.matrix_rank(np.hstack([A3, b3.reshape(-1, 1)])), ")")

    # LU reuse across multiple b's (Section 1.4)
    rhs_list = [np.array([5.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 3.0])]
    solutions = solve_via_lu_reuse(A, rhs_list)
    print("\nLU-reuse solutions for multiple right-hand sides:")
    for b_i, x_i in zip(rhs_list, solutions):
        print(f"  b={b_i} -> x={x_i}")
        assert np.allclose(A @ x_i, b_i)
    print("\nAll checks passed.")
