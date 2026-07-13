# Linear Algebra for AI/ML

A reference on the linear algebra that underpins machine learning written to sit alongside coursework and to remain useful afterward as an engineering reference.

Every topic follows the same four-part structure:

1. **Mathematical foundation** — definitions and the core proofs.
2. **Worked example** — a small numeric example carried through by hand where it clarifies the mechanics.
3. **Code** — a runnable NumPy/PyTorch implementation in [`code/`](code/), usually including a from-scratch version next to the library call.

## Contents

| # | Topic | Notes | Code |
|---|---|---|---|
| 1 | Systems of Linear Equations | [notes/01_systems_of_linear_equations.md](notes/01_systems_of_linear_equations.md) | [code/01_systems_of_linear_equations.py](code/01_systems_of_linear_equations.py) |
| 2 | Matrices and Operations | [notes/02_matrices_and_operations.md](notes/02_matrices_and_operations.md) | [code/02_matrices_and_operations.py](code/02_matrices_and_operations.py) |
| 3 | Vector Spaces | [notes/03_vector_spaces.md](notes/03_vector_spaces.md) | [code/03_vector_spaces.py](code/03_vector_spaces.py) |
| 4 | Basis and Change of Basis | [notes/04_basis_and_change_of_basis.md](notes/04_basis_and_change_of_basis.md) | [code/04_basis_and_change_of_basis.py](code/04_basis_and_change_of_basis.py) |
| 5 | Eigenvalues and Eigenvectors | [notes/05_eigenvalues_and_eigenvectors.md](notes/05_eigenvalues_and_eigenvectors.md) | [code/05_eigenvalues_and_eigenvectors.py](code/05_eigenvalues_and_eigenvectors.py) |
| 6 | Orthogonality and QR Decomposition | [notes/06_orthogonality_and_qr_decomposition.md](notes/06_orthogonality_and_qr_decomposition.md) | [code/06_qr_decomposition.py](code/06_qr_decomposition.py) |
| 7 | Singular Value Decomposition | [notes/07_singular_value_decomposition.md](notes/07_singular_value_decomposition.md) | [code/07_svd.py](code/07_svd.py) |

For a structured, resource-linked study plan (videos, courses, textbooks per topic)->(LEARNING_PATH.md).

## Prerequisites

- Basic calculus (partial derivatives, for the least-squares derivations).
- Python + NumPy fundamentals. PyTorch is used only where GPU batching / autodiff is the actual point.

## Running the code

```bash
git clone https://github.com/tahreem7926/Linear-Algebra-for-ML
cd linear-algebra-for-ml
pip install -r requirements.txt
python code/06_qr_decomposition.py   # any file runs standalone
```

Each script in `code/` is self-contained, prints its own output, and cross-checks any from-scratch implementation against the equivalent NumPy/SciPy call; if the script runs without an assertion error, the derivation in the matching notes file checks out numerically too.

## A note on proofs

Existence proofs are usually constructive (they show you *how* to build the object), because that construction is normally also the algorithm. Where a fully rigorous proof would be long and add little practical insight (e.g. uniqueness of a basis's cardinality via the Steinitz exchange lemma), the note says so explicitly and gives a reference.

## License

MIT 
