# 7. Singular Value Decomposition (SVD)

## 7.1 Statement

For **any** matrix $A \in \mathbb{R}^{m\times n}$ (square, rectangular, full-rank, or rank-deficient — no restrictions), there exist orthogonal $U\in\mathbb{R}^{m\times m}$, orthogonal $V\in\mathbb{R}^{n\times n}$, and a diagonal (in the rectangular sense — nonzero only on the leading diagonal) $\Sigma\in\mathbb{R}^{m\times n}$ with entries $\sigma_1\geq\sigma_2\geq\cdots\geq 0$ such that

$$A = U\Sigma V^\top$$

The $\sigma_i$ are the **singular values**; columns of $U$ are **left singular vectors**, columns of $V$ are **right singular vectors**. Unlike eigendecomposition (Chapter 5), which only applies to square matrices and isn't guaranteed to exist even then, **SVD always exists, for every matrix, full stop.** This universality is exactly why it's the workhorse decomposition in applied linear algebra.

## 7.2 Derivation (construction, not just an existence claim)

Start from $A^\top A \in \mathbb{R}^{n\times n}$. It's symmetric ($( A^\top A)^\top = A^\top A$) and positive semi-definite ($x^\top A^\top A x = \|Ax\|^2 \geq 0$ for all $x$). By the spectral theorem (Chapter 5), $A^\top A$ has an orthonormal eigenbasis:

$$A^\top A = V\Lambda V^\top, \qquad \Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_n),\ \lambda_i\geq 0$$

Define $\sigma_i = \sqrt{\lambda_i}$ (well-defined and real since $\lambda_i \geq 0$). For every $\sigma_i > 0$, define

$$u_i = \frac{Av_i}{\sigma_i}$$

**Claim: the $u_i$ are orthonormal.**

$$\langle u_i, u_j\rangle = \frac{(Av_i)^\top(Av_j)}{\sigma_i\sigma_j} = \frac{v_i^\top A^\top A v_j}{\sigma_i\sigma_j} = \frac{v_i^\top(\lambda_j v_j)}{\sigma_i\sigma_j} = \frac{\lambda_j}{\sigma_i\sigma_j}\langle v_i,v_j\rangle = \frac{\lambda_j}{\sigma_i\sigma_j}\delta_{ij}$$

For $i=j$: $\frac{\lambda_i}{\sigma_i^2} = \frac{\sigma_i^2}{\sigma_i^2} = 1$. For $i\neq j$: $\delta_{ij}=0$ kills the term regardless of the prefactor. So $\langle u_i,u_j\rangle=\delta_{ij}$. $\blacksquare$

If $A$ has rank $r < n$ (i.e. $\lambda_{r+1}=\cdots=\lambda_n=0$), the construction above only produces $r$ vectors $u_1,\ldots,u_r$; extend this to a full orthonormal basis $u_1,\ldots,u_m$ of $\mathbb{R}^m$ arbitrarily (always possible — any orthonormal set extends to a full orthonormal basis, by Gram-Schmidt applied to any completion). By construction, $Av_i = \sigma_i u_i$ for $i\leq r$ and $Av_i = 0$ for $i>r$ (since $\lambda_i=0 \Rightarrow \|Av_i\|^2 = v_i^\top A^\top A v_i = \lambda_i = 0$). Stacking all $n$ of these equations as columns:

$$AV = U\Sigma \implies A = U\Sigma V^\top$$

(using $V^{-1}=V^\top$, since $V$ is orthogonal). $\blacksquare$

## 7.3 Relationship to eigendecomposition

From the derivation: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$ — **singular values of $A$ are square roots of eigenvalues of $A^\top A$** (equivalently, of $AA^\top$ — same nonzero eigenvalues, standard fact). Right singular vectors $V$ = eigenvectors of $A^\top A$; left singular vectors $U$ = eigenvectors of $AA^\top$.

For $A$ itself symmetric PSD, SVD and eigendecomposition **coincide** ($U=V$, $\Sigma=\Lambda$) — SVD is a genuine generalization of eigendecomposition to non-square, non-symmetric, or rank-deficient matrices, not a different tool.

## 7.4 The four fundamental subspaces via SVD (closing the loop from Chapter 3)

With $A = U\Sigma V^\top$, rank $r$:

| Subspace | Basis |
|---|---|
| $\mathrm{col}(A)$ | $u_1,\ldots,u_r$ |
| $\mathrm{null}(A)$ | $v_{r+1},\ldots,v_n$ |
| $\mathrm{row}(A)$ | $v_1,\ldots,v_r$ |
| $\mathrm{null}(A^\top)$ | $u_{r+1},\ldots,u_m$ |

This is the cleanest possible answer to "give me orthonormal bases for all four fundamental subspaces at once, compatible with how $A$ acts" — SVD *is* that answer. $A$ maps $v_i \mapsto \sigma_i u_i$ for $i\leq r$ and kills everything in $\mathrm{span}(v_{r+1},\ldots,v_n)$ — the simplest possible description of what a linear map can do to space (stretch some orthogonal directions by nonnegative factors, crush the rest to zero).

## 7.5 Low-rank approximation (Eckart–Young theorem)

Write $A = \sum_{i=1}^{r}\sigma_i u_i v_i^\top$ (sum of rank-1 pieces, ordered by decreasing $\sigma_i$). The **rank-$k$ truncation** $A_k = \sum_{i=1}^k \sigma_i u_iv_i^\top$ ($k<r$) is the **best possible rank-$k$ approximation of $A$** in both the Frobenius and spectral norm:

$$A_k = \arg\min_{\mathrm{rank}(B)\leq k} \|A - B\|_F$$

*Intuition (not a full proof):* $\|A\|_F^2 = \sum_i \sigma_i^2$ (a standard fact — the Frobenius norm is basis-independent, and in the $U,V$ bases $A$ is diagonal with entries $\sigma_i$). Keeping the $k$ largest $\sigma_i$'s and zeroing the rest keeps the largest possible share of that sum for a given budget of $k$ nonzero terms — and it turns out (Eckart–Young, full proof via a minimax characterization of singular values) that no *other* rank-$k$ matrix, not just other diagonal truncations, can do better. The full proof is a genuine theorem, not just "obviously the biggest terms matter most" — see Trefethen & Bau or the Brunton SVD playlist (Section "Matrix Approximation") for it in full.

**This is the mathematical content behind "SVD compresses data"**: if $A$'s singular values decay fast, a small $k$ captures almost all of $\|A\|_F$, and $A_k$ is a much smaller object (only needing $k(m+n+1)$ numbers instead of $mn$) that's provably as close to $A$ as any rank-$k$ matrix can get.

## 7.6 The Moore–Penrose pseudoinverse

$$A^+ = V\Sigma^+U^\top, \qquad \Sigma^+_{ii} = \begin{cases}1/\sigma_i & \sigma_i > 0\\ 0 & \sigma_i = 0\end{cases}$$

$A^+$ generalizes matrix inversion to non-square and rank-deficient matrices, and solves least squares (Chapter 6) even when $A$ is **not** full column rank — a case QR alone doesn't handle cleanly, since $R$ becomes singular. $x = A^+b$ gives the minimum-norm solution among all $x$ minimizing $\|Ax-b\|$ (unique when a range of minimizers exists, by additionally picking the smallest one). This is the practical reason `np.linalg.lstsq` defaults to an SVD-based solver rather than QR: it degrades gracefully to the rank-deficient case instead of failing outright.

## 7.7 Where this shows up in AI/ML

- **PCA, done the numerically correct way.** Textbook PCA says "eigendecompose the covariance matrix $\Sigma = \frac1n X^\top X$" (for mean-centered data matrix $X$). In practice, SVD is applied *directly to $X$* instead: $X = U\Sigma V^\top \Rightarrow X^\top X = V\Sigma^2 V^\top$, so the right singular vectors of $X$ are exactly the eigenvectors of the covariance matrix, and $\sigma_i^2/n$ recovers the eigenvalues — **without ever forming $X^\top X$**. This is the same condition-number-squaring argument from Chapter 6 (normal equations vs. QR), applied to PCA instead of regression: forming $X^\top X$ squares $X$'s condition number, so going through SVD of $X$ directly is more numerically stable, which is exactly why `sklearn.decomposition.PCA` uses SVD internally rather than eigendecomposing a covariance matrix by hand.
- **Image / data compression**: truncated SVD of an image (viewed as a matrix of pixel intensities) is a direct application of Section 7.5 — demonstrated numerically in `code/07_svd.py`.
- **Recommender systems**: classical matrix factorization approaches to collaborative filtering (the Netflix Prize–era methods) are, at their core, a low-rank approximation of a sparse user-item ratings matrix — conceptually SVD, though production versions solve a regularized factorization objective directly (via SGD/ALS) rather than computing an exact SVD of the (incomplete!) ratings matrix.
- **Latent Semantic Analysis (NLP)**: SVD of a term-document matrix produces "topics" as singular-vector directions — an early (pre-neural) approach to semantic embeddings.
- **Denoising**: if noise is spread roughly evenly across all directions but signal concentrates in a few, truncating small singular values removes proportionally more noise than signal.
- **Condition number**, precisely defined: $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ — this is the exact quantity that was being informally invoked in Chapter 6's "$A^\top A$ has condition number $\kappa(A)^2$" claim; now it has a precise definition in terms of singular values.
- **Computing numerical rank**: count singular values above some small threshold — more robust in floating point than checking for exact linear dependence, which Chapter 3's definition assumes exact arithmetic.

## 7.8 Worked example

$$A = \begin{bmatrix}3 & 0\\0 & -2\\0&0\end{bmatrix}$$

Already (almost) in SVD form by inspection: $A^\top A = \begin{bmatrix}9&0\\0&4\end{bmatrix}$, eigenvalues $9,4$, so $\sigma_1=3,\ \sigma_2=2$, $V=I$ (eigenvectors are the standard basis already). $u_1 = Av_1/\sigma_1 = (1,0,0)$, and since the second diagonal entry of $A$ is $-2$ (negative), $u_2 = Av_2/\sigma_2 = (0,-1,0)$ — note the sign flip absorbed into $u_2$, which is exactly why singular values are defined to be **nonnegative** by convention while eigenvalues are allowed to be negative: SVD trades "sign lives on the value" for "sign lives on the vector direction." Extend $U$ with $u_3=(0,0,1)$ to complete the basis of $\mathbb{R}^3$. Full numeric construction (including the "derive it from $A^\top A$ by hand, then confirm against `np.linalg.svd`" pattern used throughout this repo) is in `code/07_svd.py`, along with a low-rank image compression demo.

## Common pitfalls

- **Assuming $U$ and $V$ from `np.linalg.svd` will match a hand derivation exactly.** Signs of singular vectors are not unique (flipping the sign of both $u_i$ and $v_i$ together leaves $\sigma_i u_iv_i^\top$ unchanged), and when singular values repeat, the corresponding subspace's basis isn't unique either. Compare via reconstruction ($U\Sigma V^\top \approx A$) rather than expecting entrywise equality with a by-hand answer.
- **Forgetting SVD's $\Sigma$ is $m\times n$, not $n\times n$.** For $m\neq n$, $\Sigma$ is rectangular with zeros padding out the non-square part — a common source of dimension-mismatch bugs when implementing truncated SVD by hand instead of using library slicing.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: MIT 18.06 Lecture 29, and the Steve Brunton SVD playlist in full (it's the deepest applied treatment listed in this repo, including the image-compression and Netflix Prize case studies referenced above).
