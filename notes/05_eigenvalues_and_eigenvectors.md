# 5. Eigenvalues and Eigenvectors

## 5.1 Definition

$v \neq 0$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ if

$$Av = \lambda v$$

Geometrically: $A$ acts on $v$ purely by scaling — no rotation off the line spanned by $v$. Most vectors get rotated *and* scaled by $A$; eigenvectors are the special directions where only scaling happens.

## 5.2 The characteristic polynomial

$Av=\lambda v \iff (A-\lambda I)v = 0$. This has a nontrivial solution $v\neq 0$ iff $A - \lambda I$ is singular, iff $\det(A-\lambda I) = 0$.

$$p(\lambda) = \det(A - \lambda I) = 0$$

is the **characteristic polynomial** — degree $n$ for an $n\times n$ matrix, so it has (counting multiplicity, and allowing complex roots) exactly $n$ eigenvalues.

*This derivation matters more than it looks:* it converts an eigenvector search (a statement about vectors) into a polynomial root-finding problem (a statement about scalars), which is why eigenvalues can be discussed before eigenvectors — you find the $\lambda$'s first (roots of $p$), then for each $\lambda$ solve the now-linear system $(A-\lambda I)v=0$ for $v$ (Chapter 1's machinery, applied to a singular matrix by construction).

## 5.3 Multiplicity and diagonalizability

For an eigenvalue $\lambda$:
- **Algebraic multiplicity** = multiplicity of $\lambda$ as a root of $p(\lambda)$.
- **Geometric multiplicity** = $\dim(\mathrm{null}(A-\lambda I))$, i.e. how many independent eigenvectors $\lambda$ actually has.

Geometric multiplicity is always $\leq$ algebraic multiplicity. When they're equal for every eigenvalue, $A$ has $n$ independent eigenvectors total and is **diagonalizable**.

**Diagonalization.** If $A$ has $n$ independent eigenvectors $v_1,\ldots,v_n$ with eigenvalues $\lambda_1,\ldots,\lambda_n$, let $P = [v_1\ \cdots\ v_n]$ and $D = \mathrm{diag}(\lambda_1,\ldots,\lambda_n)$. Then

$$A = PDP^{-1}$$

*Proof.* Column $i$ of $AP$ is $Av_i = \lambda_i v_i$, which is exactly column $i$ of $PD$. So $AP = PD$, and since $P$'s columns are independent, $P$ is invertible: $A = PDP^{-1}$. $\blacksquare$ — this is literally Chapter 4's change-of-basis formula ($A' = P^{-1}AP$) read backwards: diagonalizing $A$ *is* changing to the eigenbasis, where the map's representation is as simple as it can possibly be.

**Not every matrix is diagonalizable** — e.g. $\begin{bmatrix}0&1\\0&0\end{bmatrix}$ has the repeated eigenvalue $0$ (algebraic multiplicity 2) but only one independent eigenvector (geometric multiplicity 1). This is a genuine obstruction, not a computational inconvenience — no basis makes this matrix diagonal.

## 5.4 The spectral theorem (symmetric matrices are always this well-behaved)

**Theorem.** If $A = A^\top$ (real, symmetric), then all eigenvalues of $A$ are real, and $A$ has an orthonormal basis of eigenvectors: $A = Q\Lambda Q^\top$ with $Q$ orthogonal, $\Lambda$ diagonal.

*Proof sketch:*
- **Real eigenvalues.** For complex $\lambda$ with eigenvector $v$ (possibly complex), $v^*Av = \lambda v^*v$ where $v^*$ is the conjugate transpose. Since $A$ is real symmetric, $(v^*Av)^* = v^*A^\top v = v^*Av$ — so $v^*Av$ is real. And $v^*v = \|v\|^2$ is real and positive. So $\lambda = \frac{v^*Av}{v^*v}$ is a ratio of two real numbers, hence real.
- **Orthogonal eigenvectors for distinct eigenvalues.** Suppose $Av=\lambda v$, $Aw = \mu w$, $\lambda\neq\mu$. Then $\lambda\langle v,w\rangle = \langle Av,w\rangle = \langle v, A^\top w\rangle = \langle v, Aw\rangle = \mu\langle v,w\rangle$ (using $A^\top = A$). So $(\lambda-\mu)\langle v,w\rangle = 0$, and since $\lambda\neq\mu$, $\langle v,w\rangle = 0$. $\blacksquare$ (Equal eigenvalues need a bit more work — the eigenspace itself turns out to be big enough to always find an orthonormal basis within it; omitted here, see MIT 18.06 Lecture 25.)

**This theorem is the reason PCA works at all.** Covariance matrices are symmetric (in fact symmetric PSD, Section 2.7) by construction, so the spectral theorem *guarantees*, without any extra assumption or numerical luck, that a covariance matrix has a full orthonormal set of eigenvectors — meaning "rotate into the eigenbasis" (Chapter 4) is always a legitimate, information-preserving change of basis for any covariance matrix, of any dataset, ever. That guarantee is doing more work in the PCA derivation than most treatments let on.

## 5.5 Power iteration

A simple algorithm for the eigenvector with the **largest-magnitude** eigenvalue (the "dominant" eigenvector), without computing the full characteristic polynomial:

$$x_{k+1} = \frac{Ax_k}{\|Ax_k\|}$$

started from a random $x_0$.

*Why it converges.* Expand $x_0$ in the eigenbasis (assume $A$ diagonalizable with eigenpairs $(\lambda_i, v_i)$, sorted $|\lambda_1| > |\lambda_2| \geq \cdots$): $x_0 = \sum_i c_i v_i$. Then

$$A^k x_0 = \sum_i c_i \lambda_i^k v_i = \lambda_1^k\left(c_1 v_1 + \sum_{i>1} c_i\left(\frac{\lambda_i}{\lambda_1}\right)^k v_i\right)$$

Since $|\lambda_i/\lambda_1| < 1$ for $i>1$, every term but the first shrinks to zero as $k\to\infty$. After normalizing at each step (to prevent overflow/underflow), the direction of $x_k$ converges to $v_1$ — geometrically at rate $|\lambda_2/\lambda_1|$, so the method converges fast when the top two eigenvalues are well-separated and slowly when they're close. $\blacksquare$

## 5.6 Where this shows up in AI/ML

- **PCA**: principal components are the eigenvectors of the data covariance matrix, ranked by eigenvalue (= variance explained along that direction). See Chapter 7 for why SVD is the numerically preferred way to actually compute this.
- **PageRank**: the ranking vector is the dominant eigenvector of a (column-stochastic) link matrix, computed in practice via power iteration on a web-scale graph — exactly Section 5.5's algorithm, at enormous scale.
- **Spectral clustering**: clusters correspond to eigenvectors of a graph Laplacian associated with small eigenvalues.
- **Optimization landscape / convexity**: a quadratic loss $f(x) = \frac12 x^\top H x$ is convex iff $H$ is positive semi-definite, i.e. iff all its eigenvalues are $\geq 0$ — directly checkable via Section 5.4 since Hessians are symmetric. Near a critical point, the Hessian's eigenvalues determine whether it's a min, max, or saddle.
- **RNN/deep-net stability**: the largest eigenvalue (spectral radius) of a recurrent weight matrix governs whether repeated application causes gradients/activations to explode ($|\lambda_1|>1$) or vanish ($|\lambda_1|<1$) — a direct consequence of the power-iteration expansion in Section 5.5, applied to *any* repeated matrix application, not just an algorithm you run on purpose.

## 5.7 Worked example

$$A = \begin{bmatrix}4 & 1\\2 & 3\end{bmatrix}$$

Characteristic polynomial: $\det(A-\lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda-5)(\lambda-2)$, so $\lambda_1=5,\ \lambda_2=2$.

For $\lambda_1=5$: $(A-5I)v=0 \Rightarrow \begin{bmatrix}-1&1\\2&-2\end{bmatrix}v=0 \Rightarrow v_1 = (1,1)$.
For $\lambda_2=2$: $\begin{bmatrix}2&1\\2&1\end{bmatrix}v=0 \Rightarrow v_2=(1,-2)$.

Check: $\mathrm{tr}(A) = 7 = 5+2$ ✓ and $\det(A) = 10 = 5\times2$ ✓ (trace = sum of eigenvalues, determinant = product — a fast sanity check worth always running). Power iteration convergence to $v_1$ demonstrated in `code/05_eigenvalues_and_eigenvectors.py`.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 14, MIT 18.06 Lectures 21–22, 25.
