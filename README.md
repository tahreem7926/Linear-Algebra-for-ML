# Linear Algebra for AI/ML

A from-scratch, proof-first reference on the linear algebra that underpins machine learning — written to sit alongside coursework (originally built while working through FAST-NUCES's linear algebra sequence and Imperial College London's *Mathematics for Machine Learning* on Coursera) and to remain useful afterward as an interview and engineering reference.

Every topic follows the same four-part structure:

1. **Mathematical foundation** — definitions and the core proofs, not just statements.
2. **Why it matters in AI/ML** — the specific place this shows up (PCA, optimization, computer vision, NLP, etc.), not a generic "this is useful" gesture.
3. **Worked example** — a small numeric example carried through by hand where it clarifies the mechanics.
4. **Code** — a runnable NumPy/PyTorch implementation in [`code/`](code/), usually including a from-scratch version next to the library call so you can see they agree.

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

Recommended reading order is top to bottom — each chapter leans on the ones before it (vector spaces before basis change, eigendecomposition before SVD, Gram-Schmidt before QR, etc.).

For a structured, resource-linked study plan (videos, courses, textbooks per topic), see [LEARNING_PATH.md](LEARNING_PATH.md).

## Why this exists

Most linear algebra material picks a lane: pure-math texts prove everything but never touch a data matrix; ML blog posts wave at "eigenvectors of the covariance matrix" without ever deriving why that's the right thing to compute. The gap between "I passed the exam" and "I can explain to an interviewer *why* QR is used instead of the normal equations, and implement it" is exactly the gap this repo tries to close. Each proof is here so a claim like "SVD always exists" isn't taken on faith, and each code example is here so the proof isn't just symbol-pushing either.

## Prerequisites

- Comfort with basic calculus (partial derivatives, for the least-squares derivations).
- Python + NumPy fundamentals. PyTorch is used only where GPU batching / autodiff is the actual point (noted per-example).
- No prior linear algebra assumed beyond high-school vectors — but this moves fast, in the spirit of a second pass through the material rather than a first.

## Running the code

```bash
git clone <this-repo-url>
cd linear-algebra-for-ml
pip install -r requirements.txt
python code/06_qr_decomposition.py   # any file runs standalone
```

Each script in `code/` is self-contained, prints its own output, and cross-checks any from-scratch implementation against the equivalent NumPy/SciPy call — if the script runs without an assertion error, the derivation in the matching notes file checks out numerically too.

## A note on proofs

Proofs here are written at "convince a working engineer, don't skip the step that actually matters" depth rather than full textbook rigor — e.g. existence proofs are usually constructive (they show you *how* to build the object), because that construction is normally also the algorithm. Where a fully rigorous proof would be long and add little practical insight (e.g. uniqueness of a basis's cardinality via the Steinitz exchange lemma), the note says so explicitly and gives a reference instead of pretending the sketch is the whole story.

## License

MIT — use freely, attribution appreciated but not required.
