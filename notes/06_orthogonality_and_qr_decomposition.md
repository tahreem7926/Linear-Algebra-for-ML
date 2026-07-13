# 6. Orthogonality and QR Decomposition

This chapter gets more depth than the others by request — QR shows up constantly in ML tooling (it's what's often running underneath `np.linalg.lstsq`) but is rarely derived end-to-end in ML-facing material.

## 6.1 Orthogonal and orthonormal vectors

Vectors $u,v$ are **orthogonal** if $\langle u,v\rangle = 0$. A set is **orthonormal** if additionally every vector has unit norm. An **orthogonal matrix** $Q$ has orthonormal columns, which is equivalent to

$$Q^\top Q = I$$

**Key consequence: $Q^{-1} = Q^\top$.** Immediate from the definition — $Q^\top Q = I$ *is* the inverse relationship, for square $Q$.

**Orthogonal matrices preserve lengths and angles.** For any $x$: $\|Qx\|^2 = (Qx)^\top(Qx) = x^\top Q^\top Q x = x^\top I x = \|x\|^2$. Since inner products (and hence angles, via Cauchy–Schwarz) are determined by norms via the polarization identity, angles are preserved too. This is why orthogonal transformations are exactly the rotations and reflections — they're the maps that don't distort geometry, only reorient it.

This norm-preservation is the property that makes orthogonal matrices numerically special: applying $Q$ never amplifies rounding error, because it can't amplify any vector's length.

## 6.2 Gram-Schmidt orthogonalization

**Goal:** given linearly independent $a_1,\ldots,a_n$, construct an orthonormal set $u_1,\ldots,u_n$ spanning the *same* subspaces at every step (i.e. $\mathrm{span}(u_1,\ldots,u_k) = \mathrm{span}(a_1,\ldots,a_k)$ for every $k$).

**Construction:**

$$u_1 = \frac{a_1}{\|a_1\|}$$

For $k = 2,\ldots,n$: subtract off the projection onto every previously-built $u_i$, then normalize what's left.

$$w_k = a_k - \sum_{i=1}^{k-1} \langle a_k, u_i\rangle\, u_i, \qquad u_k = \frac{w_k}{\|w_k\|}$$

**Why this works — the orthogonal projection formula.** For a unit vector $u_i$, $\langle a_k,u_i\rangle u_i$ is the orthogonal projection of $a_k$ onto the line spanned by $u_i$ — the closest point on that line to $a_k$, and $a_k$ minus that projection is, by construction, orthogonal to $u_i$: $\langle a_k - \langle a_k,u_i\rangle u_i,\ u_i\rangle = \langle a_k,u_i\rangle - \langle a_k,u_i\rangle\langle u_i,u_i\rangle = \langle a_k,u_i\rangle - \langle a_k,u_i\rangle\cdot 1 = 0$ (using $\|u_i\|=1$).

**Proof that $u_k \perp u_j$ for all $j<k$ (by induction on $k$).** Assume $u_1,\ldots,u_{k-1}$ are already orthonormal. For any $j<k$:

$$\langle w_k, u_j\rangle = \langle a_k, u_j\rangle - \sum_{i=1}^{k-1}\langle a_k,u_i\rangle\langle u_i,u_j\rangle = \langle a_k,u_j\rangle - \langle a_k,u_j\rangle\cdot 1 = 0$$

— the sum collapses to a single term because $\langle u_i,u_j\rangle = 0$ for $i\neq j$ (inductive hypothesis) and $=1$ for $i=j$. So $w_k \perp u_j$ for every $j<k$, hence $u_k = w_k/\|w_k\| \perp u_j$ too. $\blacksquare$ ($w_k \neq 0$ is guaranteed by the linear independence of the $a_i$'s — if it were zero, $a_k$ would already be a combination of $a_1,\ldots,a_{k-1}$.)

**Same-span property (by induction):** $u_k$ is a combination of $a_k$ and $u_1,\ldots,u_{k-1}$ (hence, by the inductive hypothesis, a combination of $a_1,\ldots,a_k$), and conversely $a_k$ can be recovered from $u_1,\ldots,u_k$ by rearranging the defining equation — so the two spans coincide at every step.

## 6.3 QR decomposition

Rearranging the Gram-Schmidt construction: $a_k = \|w_k\|u_k + \sum_{i<k}\langle a_k,u_i\rangle u_i$. Define

$$r_{ik} = \langle a_k, u_i\rangle \ (i<k), \qquad r_{kk} = \|w_k\|$$

so that $a_k = \sum_{i\leq k} r_{ik}\,u_i$. In matrix form, with $A=[a_1\ \cdots\ a_n]$, $Q=[u_1\ \cdots\ u_n]$:

$$A = QR$$

where $Q$ has orthonormal columns ($Q^\top Q = I$) and $R$ is **upper triangular** with positive diagonal entries $r_{kk}=\|w_k\|>0$ (upper triangular precisely because $r_{ik}$ is only defined, i.e. only nonzero, for $i\leq k$ — a direct readout of the algorithm's structure, not an extra fact to prove).

This is the **"economy" or "reduced" QR**: for $A\in\mathbb{R}^{m\times n}$ with $m\geq n$ full column rank, $Q\in\mathbb{R}^{m\times n}$ (orthonormal columns, not a full orthogonal matrix unless $m=n$) and $R\in\mathbb{R}^{n\times n}$.

**Uniqueness.** For $A$ full column rank, the QR factorization with $R$'s diagonal entries positive is unique. *Sketch:* at each step of Gram-Schmidt, $u_k$ is determined up to sign by $a_1,\ldots,a_k$; requiring $r_{kk}=\|w_k\|>0$ pins down the sign. Since the construction is forced at every step, the whole factorization is forced.

## 6.4 Numerical reality: classical Gram-Schmidt is not what you should implement

The construction above ("classical Gram-Schmidt," CGS) is correct in exact arithmetic but **loses orthogonality in floating point** when columns of $A$ are close to parallel — rounding error in early steps compounds because every $u_k$ is computed from the *original* $a_k$ minus projections computed from (already slightly-off) earlier $u_i$'s.

**Modified Gram-Schmidt (MGS)** fixes this with a small but important reordering: instead of projecting the original $a_k$ onto all previous $u_i$ at once, subtract each projection immediately and project the *updated* vector at each subsequent step:

```
v_k = a_k
for i in 1..k-1:
    v_k = v_k - <v_k, u_i> u_i     # project out u_i from the running vector, not from the original a_k
u_k = v_k / ||v_k||
```

Mathematically identical to CGS in exact arithmetic; meaningfully more stable in floating point because each subtraction works with the most up-to-date (already partially-orthogonalized) vector rather than compounding error from the raw original.

**Householder reflections** go further and are what production libraries (LAPACK's `dgeqrf`, and hence NumPy/SciPy's `qr`) actually use — they build $Q^\top$ as a product of reflection matrices $H_k = I - 2\frac{vv^\top}{v^\top v}$ that each zero out everything below the diagonal in one column at a time, and are backward stable (the computed factorization is the *exact* factorization of a matrix very close to $A$) in a way Gram-Schmidt, even modified, isn't quite. The mechanics are out of scope for this note; see Trefethen & Bau, *Numerical Linear Algebra*, Lectures 7–8, for the full derivation.

**Practical takeaway:** know Gram-Schmidt because it's where the *meaning* of QR comes from (it's an orthogonalization of the columns, one at a time); use `np.linalg.qr` (Householder-based) in real code.

## 6.5 Least squares via QR

**Problem.** For $A\in\mathbb{R}^{m\times n}$, $m>n$, full column rank (an overdetermined system — more equations than unknowns, generically inconsistent), find

$$\hat x = \arg\min_x \|Ax-b\|^2$$

### The normal equations (and why they're numerically risky)

Setting the gradient of $\|Ax-b\|^2 = (Ax-b)^\top(Ax-b)$ to zero:

$$\nabla_x \|Ax-b\|^2 = 2A^\top(Ax-b) = 0 \implies A^\top A\,x = A^\top b$$

giving $\hat x = (A^\top A)^{-1}A^\top b$. This is correct — but $A^\top A$ has **condition number $\kappa(A)^2$** (squared relative to $A$'s own condition number), because forming $A^\top A$ squares the singular values of $A$ (Chapter 7 makes this precise: $\sigma_i(A^\top A) = \sigma_i(A)^2$). An ill-conditioned $A$ becomes catastrophically ill-conditioned in $A^\top A$, and you lose accuracy before you even start solving the system.

### The QR approach

Factor $A = QR$ (economy QR, $Q\in\mathbb{R}^{m\times n}$ orthonormal columns, $R\in\mathbb{R}^{n\times n}$ upper triangular, invertible since $A$ is full column rank).

Extend $Q$ to a full orthogonal matrix $Q_{\text{full}} = [Q\ |\ Q_\perp] \in \mathbb{R}^{m\times m}$ (fill in the remaining $m-n$ orthonormal directions — always possible). Since $Q_{\text{full}}$ is orthogonal, it preserves norms (Section 6.1):

$$\|Ax-b\|^2 = \|Q_{\text{full}}^\top(Ax-b)\|^2 = \left\|\begin{bmatrix}Q^\top\\Q_\perp^\top\end{bmatrix}(QRx-b)\right\|^2 = \left\|\begin{bmatrix}Rx - Q^\top b\\ -Q_\perp^\top b\end{bmatrix}\right\|^2$$

using $Q^\top Q = I$ and $Q_\perp^\top Q = 0$ (orthogonal blocks). This splits into two independent terms:

$$\|Ax-b\|^2 = \|Rx - Q^\top b\|^2 + \|Q_\perp^\top b\|^2$$

**The second term doesn't depend on $x$ at all.** So the minimum over $x$ is achieved exactly when the first term is driven to zero:

$$Rx = Q^\top b$$

Since $R$ is upper triangular (and invertible), this is solved by **back-substitution** in $O(n^2)$ — no matrix inversion needed. $\blacksquare$

**This is the same answer as the normal equations, computed more stably.** Substituting $A=QR$ into $\hat x = (A^\top A)^{-1}A^\top b$: $A^\top A = R^\top Q^\top Q R = R^\top R$ (using $Q^\top Q=I$), so $\hat x = (R^\top R)^{-1}R^\top Q^\top b = R^{-1}(R^\top)^{-1}R^\top Q^\top b = R^{-1}Q^\top b$ — exactly "solve $Rx=Q^\top b$." Same formula, but the QR route never explicitly forms $A^\top A$ (avoiding the condition-number squaring) and never explicitly inverts anything (avoiding a separate source of numerical error).

**Complexity:** both approaches are $O(mn^2)$ — QR isn't asymptotically slower, it's just more numerically trustworthy for the same cost order, which is why it's the default in production numerical software.

**The residual is orthogonal to the column space** — a useful geometric fact that falls out for free: at the optimum, $b - A\hat x$ is exactly the $Q_\perp^\top b$ component from the derivation above, which lives entirely outside $\mathrm{col}(A) = \mathrm{col}(Q)$. This matches the intuition that the best approximation of $b$ within a subspace is its orthogonal projection onto that subspace, with the leftover error perpendicular to it.

## 6.6 Where this shows up in AI/ML

- **Linear regression solvers.** `numpy.linalg.lstsq` and `scikit-learn`'s `LinearRegression` don't form $(X^\top X)^{-1}X^\top y$ — they use QR or SVD (Chapter 7) internally for exactly the stability reason above. If you've ever wondered why a "textbook" implementation of the normal equations sometimes disagrees with `sklearn` on an ill-conditioned dataset, this is why.
- **Orthogonal weight initialization** in RNNs (and some CNN/Transformer variants) initializes weight matrices to be orthogonal (via QR of a random Gaussian matrix) specifically because Section 6.1's norm-preservation property mitigates vanishing/exploding activations and gradients through repeated multiplication — directly connected to the power-iteration/spectral-radius discussion in Chapter 5.
- **The QR algorithm for eigenvalues** (a different use of QR from the one in this chapter — don't conflate the two): repeatedly factor $A_k = Q_kR_k$ and set $A_{k+1}=R_kQ_k$; this iteration converges to a triangular (or block-triangular) matrix revealing the eigenvalues on the diagonal. This is what `np.linalg.eig` uses under the hood for general matrices.
- **Gram-Schmidt / orthogonalization in general** shows up any time you need to remove redundant directions from a set of vectors — e.g. orthogonalizing basis functions in Gaussian processes, or decorrelating a set of features before ridge/LASSO to make regularization strength more interpretable.

## 6.7 Worked example

$$A = \begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}$$

$a_1=(1,1,0)$, $\|a_1\|=\sqrt2$, $u_1 = (1/\sqrt2,\,1/\sqrt2,\,0)$.

$a_2=(0,1,1)$. $\langle a_2,u_1\rangle = 1/\sqrt2$. $w_2 = a_2 - \frac{1}{\sqrt2}u_1 = (0,1,1) - (1/2,1/2,0) = (-1/2,1/2,1)$. $\|w_2\| = \sqrt{1/4+1/4+1} = \sqrt{3/2}$.

$$Q = \begin{bmatrix}1/\sqrt2 & -1/\sqrt6\\ 1/\sqrt2 & 1/\sqrt6\\ 0 & 2/\sqrt6\end{bmatrix}, \qquad R = \begin{bmatrix}\sqrt2 & 1/\sqrt2\\ 0 & \sqrt{3/2}\end{bmatrix}$$

(second column of $Q$ renormalized from $w_2$: $\sqrt{3/2}=\sqrt6/2$, giving the $1/\sqrt6$ entries above). Verify $QR=A$ and $Q^\top Q = I$ numerically, and cross-check against `np.linalg.qr`, in `code/06_qr_decomposition.py` — the same script also solves a small least-squares problem both via the normal equations and via QR and shows they agree, then constructs a deliberately ill-conditioned example where the normal-equations route visibly loses accuracy and the QR route doesn't.

## Common pitfalls

- **Forming $A^\top A$ "because the normal equations are simpler to write."** They're simpler on paper; they're the numerically worse choice in code the moment $A$ is even mildly ill-conditioned. Default to `lstsq` / QR / SVD-based solvers.
- **Confusing the two uses of "QR" in the literature** — QR *decomposition* (this chapter, a one-shot factorization) vs. the QR *algorithm* (Section 6.6, an iterative eigenvalue method that happens to use QR decomposition as its inner step). They share a name, not a purpose.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 9, MIT 18.06 Lecture 17, and Trefethen & Bau, *Numerical Linear Algebra*, Lectures 7–8 for Householder reflections in full.
