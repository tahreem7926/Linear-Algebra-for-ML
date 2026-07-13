# Learning Path

A study plan to pair with this repo, organized so each stage feeds the next. The pattern per topic: **watch the geometric intuition first, then read the proof-heavy notes here, then do the corresponding Coursera module for the guided coding practice.**

## Core spine (do these in order, once)

| Stage | Resource | What it's for |
|---|---|---|
| 1 | [3Blue1Brown — *Essence of Linear Algebra*](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (16 videos, ~3 hrs total) | Geometric intuition before any of the symbol-pushing. If a proof in this repo feels unmotivated, the matching 3Blue1Brown chapter is almost always the fix. |
| 2 | Imperial College London — [*Mathematics for Machine Learning: Linear Algebra*](https://www.coursera.org/learn/linear-algebra-machine-learning) (Coursera) | The course you're already in — covers systems of equations → matrices → basis change → eigenvalues, with graded Jupyter labs. Use this repo's notes as the "why does this step work" companion when a lecture moves fast. |
| 3 | Gilbert Strang — [MIT 18.06, *Linear Algebra*](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (OCW, full lecture videos + problem sets) | The rigor pass. Strang's framing of the four fundamental subspaces is the single most useful mental model in this entire repo — worth watching even if you only do Lectures 1–10 and 21–25 (eigenvalues, SVD). |
| 4 | Imperial College London — [*Mathematics for Machine Learning: PCA*](https://www.coursera.org/learn/pca-machine-learning) (Coursera) | Do this **after** chapters 4–7 of this repo (basis change, eigenvectors, orthogonality, SVD) — it assumes all four and is the natural capstone, deriving PCA from orthogonal projection instead of quoting it. |
| 5 | Steve Brunton — [*Singular Value Decomposition*](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv) (playlist, based on Brunton & Kutz, *Data-Driven Science and Engineering*) | The applied deep-dive for chapters 6–7: least squares via SVD, image compression, eigenfaces, the Netflix Prize matrix-completion story, randomized SVD. This is where the "why do practitioners actually reach for SVD" question gets answered. |

## Per-topic quick reference

| Topic in this repo | If the notes move too fast | If you want more rigor |
|---|---|---|
| 01 — Systems of equations | 3B1B Ch. 1–3 | MIT 18.06 Lectures 1–3 |
| 02 — Matrices & operations | 3B1B Ch. 3–7 | MIT 18.06 Lectures 1–2, 5 |
| 03 — Vector spaces | 3B1B Ch. 2, 7, 16 | MIT 18.06 Lectures 6, 9–10 |
| 04 — Basis & change of basis | 3B1B Ch. 13 (*Change of basis*) | MIT 18.06 Lecture 31 (*Change of basis, image compression*) |
| 05 — Eigenvalues & eigenvectors | 3B1B Ch. 14 (*Eigenvectors and eigenvalues*) | MIT 18.06 Lectures 21–22, 25 (symmetric matrices) |
| 06 — Orthogonality & QR | 3B1B Ch. 9 (*Dot products and duality*) | MIT 18.06 Lecture 17 (*Orthogonal matrices, Gram-Schmidt*); Trefethen & Bau, *Numerical Linear Algebra*, Lectures 7–8 for Householder |
| 07 — SVD | MIT 18.06 Lecture 29 (*Singular value decomposition*) | Brunton SVD playlist in full — it's the deepest treatment listed here |

## Textbooks (for reference, not required)

- Gilbert Strang, *Introduction to Linear Algebra* (5th ed.) — companion text to MIT 18.06, the most ML-friendly of the standard textbooks.
- Trefethen & Bau, *Numerical Linear Algebra* — the standard reference for why QR/Householder is numerically preferred over the normal equations; short and dense, worth owning once you're past the basics.
- Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (free PDF, mml-book.github.io) — written by (some of) the same team behind the Coursera specialization; a good single-volume reference that mirrors this repo's ML-first angle.

## Suggested pace

This repo is written assuming you already have a first pass through the material (matches your situation — FAST-NUCES coursework + the Coursera specialization in progress). A reasonable second-pass schedule:

- **Week 1:** Chapters 1–3 (systems, matrices, vector spaces) + Coursera Linear Algebra Weeks 1–2.
- **Week 2:** Chapter 4 (basis change) + Coursera Linear Algebra Weeks 3–4. This is also a good week to revisit the shadow-projection / change-of-basis exercises if you did them earlier — the matrix-construction pattern is identical.
- **Week 3:** Chapters 5–6 (eigenvalues, QR) + start MIT 18.06 Lectures 21–22, 17.
- **Week 4:** Chapter 7 (SVD) + Coursera PCA course + Brunton SVD playlist.

Don't treat this as a hard schedule — the point of doing a second pass is to go slower on whatever didn't stick the first time, not to race through everything.
