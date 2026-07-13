# 3. Vector Spaces

## 3.1 The axioms (and why they're worth reading once)

A vector space over $\mathbb{R}$ is a set $V$ with addition and scalar multiplication satisfying: closure under both operations, associativity and commutativity of addition, existence of a zero vector and additive inverses, distributivity of scalar multiplication over vector and scalar addition, and $1\cdot v = v$.

The point of the axiom list isn't to memorize it — it's that **anything satisfying these axioms gets every theorem in this repo for free**, whether or not it "looks like" a column of numbers. Polynomials of degree $\leq n$, $m\times n$ matrices, continuous functions on $[0,1]$, and grayscale images all form vector spaces. This is why "the space of functions a neural network can represent" or "the space of kernel feature maps" can be reasoned about with the exact same machinery as $\mathbb{R}^n$ — they satisfy the same axioms.

## 3.2 Subspaces

$U \subseteq V$ is a **subspace** if it's nonempty and closed under addition and scalar multiplication.

*Why this is a complete test.* Closure under scalar multiplication with scalar $0$ forces $0 \in U$ (so you get the zero vector for free once you know $U$ is nonempty and closed under scaling). Closure under addition and scaling together give you every other axiom automatically, since $U$ inherits them from $V$. So checking two conditions verifies the whole eight-axiom list.

## 3.3 Span and linear independence

$\mathrm{span}(v_1,\ldots,v_k) = \{c_1v_1 + \cdots + c_kv_k : c_i \in \mathbb{R}\}$ — always a subspace (closed under addition/scaling by construction).

Vectors $v_1,\ldots,v_k$ are **linearly independent** if $c_1v_1+\cdots+c_kv_k = 0 \implies c_1=\cdots=c_k=0$. Otherwise they're dependent — meaning at least one vector is redundant, expressible as a combination of the others.

**Theorem.** Any set of more than $n$ vectors in $\mathbb{R}^n$ is linearly dependent.

*Proof sketch.* Stack $k>n$ vectors as columns of a matrix $A \in \mathbb{R}^{n\times k}$. The homogeneous system $Ac=0$ has more unknowns ($k$) than equations ($n$), so by the rank argument in Chapter 1 there must be a free variable — a nontrivial solution $c\neq 0$ exists, which is exactly a dependence relation. $\blacksquare$

This is the algebraic reason "you can't have more than $n$ independent directions in $n$-dimensional space" — it isn't a geometric intuition you have to take on faith, it falls directly out of the rank-counting argument from Chapter 1.

## 3.4 Basis and dimension

A **basis** of $V$ is a linearly independent set that spans $V$. Every vector in $V$ then has a *unique* representation as a combination of basis vectors (uniqueness follows from independence: two different representations would subtract to a nontrivial dependence relation).

**Theorem (well-defined dimension).** Every basis of a finite-dimensional $V$ has the same number of vectors, called $\dim(V)$.

*Proof sketch (Steinitz exchange lemma).* If $\{u_1,\ldots,u_m\}$ spans $V$ and $\{v_1,\ldots,v_k\}$ is independent in $V$, then $k \leq m$ — you can swap in the $v_i$'s for $u_j$'s one at a time without breaking the spanning property, and you run out of $u_j$'s to swap out before (or exactly when) you run out of $v_i$'s. Applying this both ways to two bases forces them to have equal size. Full details: Strang, *Introduction to Linear Algebra*, or MIT 18.06 Lecture 9 — the sketch above is the idea; the induction is mechanical but not illuminating enough to reproduce in full here.

## 3.5 Rank–nullity theorem

For $A \in \mathbb{R}^{m\times n}$, viewed as a map $\mathbb{R}^n \to \mathbb{R}^m$:

$$n = \mathrm{rank}(A) + \mathrm{nullity}(A)$$

where $\mathrm{rank}(A) = \dim(\mathrm{col}(A))$ and $\mathrm{nullity}(A) = \dim(\mathrm{null}(A))$.

*Proof sketch.* Take a basis $\{v_1,\ldots,v_k\}$ of $\mathrm{null}(A)$ ($k = \mathrm{nullity}(A)$) and extend it to a basis $\{v_1,\ldots,v_k,v_{k+1},\ldots,v_n\}$ of $\mathbb{R}^n$ (always possible — any independent set extends to a basis). Claim: $\{Av_{k+1},\ldots,Av_n\}$ is a basis for $\mathrm{col}(A)$. It spans $\mathrm{col}(A)$ because any $Ax$ can be written in terms of the $v_i$-basis and the null-space components vanish under $A$; it's independent because a dependence relation among the images would pull back (via the extension being a basis) to a dependence relation involving null-space vectors, contradicting how the basis was built. So $\mathrm{rank}(A) = n-k$, i.e. $n = \mathrm{rank}(A)+k$. $\blacksquare$

**Practical reading:** every dimension of input either gets mapped to something in the output (contributing to rank) or gets crushed to zero (contributing to nullity) — there's no third option. This is the precise version of "a matrix can't create information out of nowhere."

## 3.6 The four fundamental subspaces

For $A \in \mathbb{R}^{m\times n}$:

| Subspace | Definition | Lives in | Dimension |
|---|---|---|---|
| Column space $\mathrm{col}(A)$ | span of columns of $A$ | $\mathbb{R}^m$ | $r = \mathrm{rank}(A)$ |
| Null space $\mathrm{null}(A)$ | $\{x : Ax=0\}$ | $\mathbb{R}^n$ | $n-r$ |
| Row space $\mathrm{row}(A) = \mathrm{col}(A^\top)$ | span of rows of $A$ | $\mathbb{R}^n$ | $r$ |
| Left null space $\mathrm{null}(A^\top)$ | $\{y : A^\top y = 0\}$ | $\mathbb{R}^m$ | $m-r$ |

**Orthogonality relations** (proved properly once you have Chapter 6's inner-product tools, stated here because the *picture* matters now): $\mathrm{row}(A) \perp \mathrm{null}(A)$ — if $Ax=0$ then every row of $A$ dotted with $x$ is zero, i.e. $x$ is orthogonal to every row. Similarly $\mathrm{col}(A) \perp \mathrm{null}(A^\top)$. Together these say every vector in $\mathbb{R}^n$ splits cleanly into a row-space part and a null-space part, and every vector in $\mathbb{R}^m$ splits into a column-space part and a left-null-space part. This picture is the backbone of the SVD derivation in Chapter 7 — SVD is, among other things, the statement that there exist orthonormal bases for these four subspaces that $A$ maps to each other in the simplest possible way.

## 3.7 Inner product spaces

An inner product $\langle\cdot,\cdot\rangle$ adds geometry (lengths, angles) to a vector space: symmetric ($\langle u,v\rangle = \langle v,u\rangle$), bilinear, and positive-definite ($\langle v,v\rangle > 0$ for $v\neq 0$). The standard dot product $\langle u,v\rangle = u^\top v$ is the canonical example, but kernel methods (SVMs, Gaussian processes) work by defining *other* inner products implicitly via a kernel function $k(x,y) = \langle\phi(x),\phi(y)\rangle$, without ever computing $\phi$ explicitly — the "kernel trick."

Induced norm: $\|v\| = \sqrt{\langle v,v\rangle}$.

**Cauchy–Schwarz inequality.** $|\langle u,v\rangle| \leq \|u\|\|v\|$.

*Proof.* For any $t\in\mathbb{R}$, $\|u - tv\|^2 \geq 0$ (positive-definiteness). Expand: $\|u\|^2 - 2t\langle u,v\rangle + t^2\|v\|^2 \geq 0$. This is a quadratic in $t$ that's always nonnegative, so its discriminant must be $\leq 0$: $4\langle u,v\rangle^2 - 4\|u\|^2\|v\|^2 \leq 0$, i.e. $\langle u,v\rangle^2 \leq \|u\|^2\|v\|^2$. Take square roots. $\blacksquare$

Cauchy–Schwarz is what makes $\cos\theta = \frac{\langle u,v\rangle}{\|u\|\|v\|}$ well-defined (the ratio is guaranteed to lie in $[-1,1]$) — this is the "cosine similarity" used everywhere in ML for comparing embeddings.

## 3.8 Where this shows up in AI/ML

- **Feature spaces and embedding spaces** are vector spaces by construction; "nearby in embedding space" is a statement about the norm/inner product on that space.
- **Multicollinearity in regression** is exactly linear *dependence* (or near-dependence) among feature columns — Section 3.3's theorem is the reason a design matrix with more (correlated) features than independent directions produces an ill-conditioned or singular $A^\top A$.
- **Kernel methods** operate entirely in the language of inner-product spaces, often infinite-dimensional ones, without ever materializing a basis.
- **Function spaces**: a neural network's hypothesis space is a (highly non-linear, but locally-linear-in-pieces) subset of a function space; the vector-space viewpoint is what makes "distance between two models" or "linear interpolation between two networks' weights" meaningful operations.

## Common pitfalls

- **Treating "spans" and "is a basis for" as interchangeable.** A spanning set can be redundant (contain dependent vectors); a basis is a spanning set with the redundancy removed. If you're told a set "spans" a space, don't assume you can use it as a coordinate system until you've confirmed independence.
- **Confusing rank-nullity with "rank + nullity = number of rows."** It's the number of *columns* ($n$, the input dimension) — an easy off-by-transpose error when reading code that stores matrices with samples as rows.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 2, 7, 16, MIT 18.06 Lectures 6, 9–10.
