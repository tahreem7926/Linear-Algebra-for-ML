# 2. Matrices and Operations

## 2.1 Matrices as linear maps

A matrix $A \in \mathbb{R}^{m\times n}$ is not just a grid of numbers — it *is* a linear map $T_A:\mathbb{R}^n \to \mathbb{R}^m$, $T_A(x) = Ax$, satisfying

$$T_A(\alpha x + \beta y) = \alpha T_A(x) + \beta T_A(y).$$

Every linear map between finite-dimensional spaces can be represented this way once you fix a basis (Chapter 4 makes this precise), and every $m\times n$ matrix defines one. This equivalence — matrix $\leftrightarrow$ linear map — is the single most load-bearing idea in the subject: it's what makes "multiply matrices" and "compose transformations" the same operation.

## 2.2 Matrix multiplication

$$(AB)_{ik} = \sum_j A_{ij}B_{jk}$$

**This definition is not arbitrary — it's forced by composition.** If $T_A(x) = Ax$ and $T_B(x) = Bx$, then

$$(T_A \circ T_B)(x) = A(Bx) = (AB)x$$

so matrix multiplication is defined exactly so that it represents function composition. Everything else (associativity, non-commutativity) follows from that.

**Associativity proof.** 

$$
\begin{aligned}
((AB)C)_{il} &= \sum_k (AB)_{ik}C_{kl} \\
&= \sum_k \sum_j A_{ij}B_{jk}C_{kl} \\
&= \sum_j A_{ij}\sum_k B_{jk}C_{kl} \\
&= \sum_j A_{ij}(BC)_{jl} \\
&= (A(BC))_{il}
\end{aligned}
$$

The sums commute freely because they're finite sums of real numbers, so the order of summation doesn't matter — the "proof" is really just relabeling which sum you do first.

**Non-commutativity is not a technicality — it's geometry.** $AB \neq BA$ in general because "rotate then shear" is a different transformation from "shear then rotate." Composition of functions was never commutative; matrix multiplication inherits that.

## 2.3 Transpose and symmetric matrices

$$
\begin{aligned}
((AB)^\top)_{ij} &= (AB)_{ji} \quad \blacksquare
\end{aligned}
$$. Key identity: $(AB)^\top = B^\top A^\top$ (order reverses).

*Proof.*
$$
\begin{aligned}
((AB)^\top)_{ij} &= (AB)_{ji} \\
&= \sum_k A_{jk}B_{ki} \\
&= \sum_k B_{ki}A_{jk} \\
&= \sum_k (B^\top)_{ik}(A^\top)_{kj} \\
&= (B^\top A^\top)_{ij} \quad \blacksquare
\end{aligned}
$$

$A$ is **symmetric** if $A^\top = A$. Symmetric matrices are exactly the matrices that arise from quadratic forms and inner-product-like structures — covariance matrices, Gram matrices, and Hessians of scalar functions are all symmetric, which is why Chapter 5's spectral theorem (special eigenstructure for symmetric matrices) is quietly doing most of the work behind PCA, kernel methods, and second-order optimization.

## 2.4 Inverses

$A^{-1}$ satisfies $A^{-1}A = AA^{-1} = I$. Exists iff $\det(A)\neq 0$ iff $A$ has full rank iff the columns of $A$ are linearly independent (Chapter 3).

**Uniqueness proof.** Suppose $B$ and $C$ are both inverses of $A$. Then $B = BI = B(AC) = (BA)C = IC = C$. So if an inverse exists, it's the only one. $\blacksquare$ (This is a standard "identity-sandwich" trick — worth remembering the pattern, it reappears constantly.)

**$(AB)^{-1} = B^{-1}A^{-1}$** (note the order reversal, same reason as the transpose): check $(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = I$.

Computing $A^{-1}$ explicitly is rarely the right move numerically — solving $Ax=b$ via elimination or QR (Chapter 6) is cheaper and more stable than forming $A^{-1}$ and multiplying. If you find yourself writing `np.linalg.inv(A) @ b` in real code, `np.linalg.solve(A, b)` is almost always the better call — same mathematical answer, meaningfully better numerics and roughly 3x fewer flops.

## 2.5 Determinant

For a $2\times2$ matrix, $\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad-bc$. In general, the determinant is defined recursively by cofactor expansion, but the definition worth internalizing is geometric:

**$|\det(A)|$ is the factor by which $A$ scales volume.** The unit square (cube, hypercube) has volume 1; its image under $A$ has volume $|\det(A)|$. This is why $\det(A) = 0$ exactly characterizes singularity: it means $A$ collapses $n$-dimensional volume to zero, i.e., it squashes space into a lower-dimensional subspace, which means the columns are linearly dependent and the map isn't invertible.

**Key properties** (stated, proofs by direct computation or via the volume interpretation):
- $\det(AB) = \det(A)\det(B)$ — "scale by A's factor, then by B's factor" composes multiplicatively, matching volume scaling under composed maps.
- $\det(A^\top) = \det(A)$.
- $\det(A^{-1}) = 1/\det(A)$ (immediate from the multiplicative property applied to $AA^{-1}=I$).
- Swapping two rows flips the sign; this is where "signed volume" (orientation) enters.

## 2.6 Trace

$\mathrm{tr}(A) = \sum_i A_{ii}$. Useful identity: $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ even though $AB \neq BA$ in general.

*Proof.* 
$$
\begin{aligned}
\mathrm{tr}(AB) &= \sum_i (AB)_{ii} \\
&= \sum_i \sum_j A_{ij}B_{ji} \\
&= \sum_j \sum_i B_{ji}A_{ij} \\
&= \sum_j (BA)_{jj} \\
&= \mathrm{tr}(BA) \quad \blacksquare
\end{aligned}
$$

This "cyclic property" allows you to rotate the matrices inside the trace:
$$
\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)
$$

This "cyclic property" ($\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$) is used constantly to simplify gradients of matrix expressions in ML derivations (e.g., deriving the gradient of $\mathrm{tr}(A^\top B)$ with respect to $A$).

## 2.7 Special matrices worth naming

| Matrix | Definition | Where it shows up |
|---|---|---|
| Identity $I$ | $I_{ij}=1$ if $i=j$ else $0$ | Multiplicative identity; residual connections in ResNets are literally "$I + $ something" |
| Diagonal | zero off-diagonal | Eigendecomposition target; efficient to store/multiply |
| Orthogonal | $Q^\top Q = QQ^\top = I$ | Rotations/reflections; preserves lengths and angles — see Chapter 6 |
| Symmetric | $A^\top = A$ | Covariance, Gram, Hessian matrices |
| Positive (semi-)definite | $x^\top A x > 0\ (\geq 0)$ for all $x\neq 0$ | Covariance matrices (always PSD), convexity of a quadratic loss (Hessian PD $\Rightarrow$ local min) |

## 2.8 Where this shows up in AI/ML

- **Every layer of a neural network** is (affine map) = matrix multiply + bias, composed via the multiplication rule above; backprop through a stack of layers is the chain rule applied to a composition of linear (and nonlinear) maps.
- **Covariance and Gram matrices** ($\Sigma = \frac{1}{n}X^\top X$ for centered data $X$, or $K_{ij} = \langle \phi(x_i), \phi(x_j)\rangle$ for kernel methods) are symmetric PSD by construction — $x^\top \Sigma x = \frac{1}{n}\|Xx\|^2 \geq 0$.
- **Convolutions as matrix multiplication**: a convolution can be unrolled into a (sparse, structured) matrix multiply via `im2col` — this is literally how some conv implementations are executed on hardware without dedicated conv kernels.
- **Batched matrix multiplication on GPUs** is the operation that dominates both training and inference cost in deep learning; understanding `einsum` and broadcasting rules (below) is a practical skill, not just theory.
- **Hessian positive-definiteness** determines whether a critical point of a loss function is a local minimum — directly using the PD definition above.

## 2.9 Worked example — composition

$$
A = \begin{bmatrix}0&-1\\1&0\end{bmatrix} \text{ (90° rotation)}, \quad
B = \begin{bmatrix}2&0\\0&1\end{bmatrix} \text{ (stretch x by 2)}
$$

$$
AB = \text{"stretch then rotate"} = \begin{bmatrix} 0 & -1 \\ 2 &  0 \end{bmatrix}
$$

$$
BA = \text{"rotate then stretch"} = \begin{bmatrix} 0 & -2 \\ 1 &  0 \end{bmatrix}
$$. Different matrices — confirming non-commutativity is geometric, not a notational accident. Verified numerically in `code/02_matrices_and_operations.py`.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 3–7, MIT 18.06 Lectures 1–2 and 5.
