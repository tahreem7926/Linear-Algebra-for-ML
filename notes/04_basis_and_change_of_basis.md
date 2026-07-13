# 4. Basis and Change of Basis

## 4.1 Coordinates relative to a basis

Given a basis $\mathcal{B} = \{b_1,\ldots,b_n\}$ of $V$, every $v\in V$ has a unique expansion $v = c_1b_1+\cdots+c_nb_n$. The tuple $[v]_\mathcal{B} = (c_1,\ldots,c_n)$ is $v$'s **coordinate vector** relative to $\mathcal{B}$.

The vector $v$ is a single geometric object; $[v]_\mathcal{B}$ is a *description* of it that depends on your choice of basis. This distinction — object vs. its coordinates in some basis — is the entire content of this chapter.

## 4.2 The change-of-basis matrix

Let $\mathcal{B}_{\text{old}} = \{e_1,\ldots,e_n\}$ (say, the standard basis) and $\mathcal{B}_{\text{new}} = \{b_1,\ldots,b_n\}$. Build

$$P = \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix}$$

— the matrix whose columns are the new basis vectors, expressed in old-basis coordinates. Then for any vector,

$$x_{\text{old}} = P\, x_{\text{new}}, \qquad x_{\text{new}} = P^{-1}x_{\text{old}}$$

*Why this is exactly the matrix-times-coordinates equation and nothing more:* $x_{\text{new}} = (c_1,\ldots,c_n)$ means $x = c_1b_1+\cdots+c_nb_n$ by definition of coordinates. Written as a matrix product, that sum is exactly $Px_{\text{new}}$. $P$ is invertible because its columns (a basis) are linearly independent by definition — this is worth flagging explicitly, because it's the fact that separates this chapter from the shadow/projection example below.

## 4.3 How a linear transformation's matrix changes with the basis

If a linear map $T$ is represented by matrix $A$ in the old basis, its representation $A'$ in the new basis is

$$A' = P^{-1}AP$$

*Derivation.* Take $x_{\text{old}} = Px_{\text{new}}$. In the old basis, $T$ acts as $y_{\text{old}} = Ax_{\text{old}} = APx_{\text{new}}$. Convert the result back to new-basis coordinates: $y_{\text{new}} = P^{-1}y_{\text{old}} = P^{-1}APx_{\text{new}}$. So $A' = P^{-1}AP$. $\blacksquare$

Matrices related this way ($A' = P^{-1}AP$ for some invertible $P$) are called **similar**, and similarity is the formal statement of "same linear map, different coordinate system." Similar matrices always share the same eigenvalues, trace, and determinant (Chapter 5) — those quantities are properties of the underlying map, not of any particular coordinate description of it.

## 4.4 Orthonormal bases: the computationally cheap special case

If $\mathcal{B}_{\text{new}}$ is **orthonormal**, $P$ is an orthogonal matrix, so $P^{-1} = P^\top$ (proved in Chapter 6). Changing basis and changing back becomes a transpose instead of a matrix inversion — the difference between an $O(n^2)$ operation and an $O(n^3)$ one for large $n$, and numerically exact rather than subject to inversion's conditioning issues. This is precisely why PCA (Chapter 7) insists on an *orthonormal* eigenbasis of the covariance matrix rather than just any old eigenbasis: it buys you $P^{-1}=P^\top$ for free.

## 4.5 A pitfall worth naming explicitly: change of basis vs. projection

It's tempting, after working through a problem like "find the matrix that projects a 3D point onto its shadow on the ground plane," to file it under "change of basis." **It isn't one, and the distinction matters:**

- A change-of-basis matrix $P$ is square and invertible — it's a relabeling of the *same* vector, and you can always recover $x_{\text{old}}$ from $x_{\text{new}}$.
- A projection (like the shadow map $A: \mathbb{R}^3 \to \mathbb{R}^2$ from the earlier exercise) is generally non-square and **not invertible** — it's rank-deficient by construction (its third row is identically zero, as derived there), because it genuinely discards information (how far above the ground the point was). You cannot recover the original 3D point from its shadow alone.

Both are linear maps, and both are represented by matrices — but "change of basis" specifically means *re-describing* a vector without losing information, while a projection *destroys* a dimension on purpose. If you catch yourself calling a rank-deficient map a "change of basis," that's the signal to check whether the matrix is even square, let alone invertible.

## 4.6 Where this shows up in AI/ML

- **PCA is literally a change of basis** — rotate the data into the eigenbasis of its covariance matrix (Chapter 5), where the new coordinate axes are ranked by variance explained. "Dimensionality reduction" is then this change of basis *followed by* a genuine projection (dropping the low-variance coordinates) — so a full PCA pipeline uses both concepts from this chapter, back to back, and it's worth being precise about which step is which.
- **Whitening transformations** (decorrelating features before feeding them to a model sensitive to scale, e.g. some classical clustering or optimization methods) are a change of basis into the covariance eigenbasis followed by rescaling each axis to unit variance.
- **JPEG/image compression** changes basis from pixel-space to the discrete cosine transform (DCT) basis, where natural images concentrate their energy in a few low-frequency coefficients — compression then drops (projects out) the small high-frequency coefficients, again a change of basis followed by a projection.
- **Computer graphics / vision**: camera-to-world and world-to-camera transforms, and transforming a 3D scene into a different coordinate frame (the shadow-projection exercise is a simplified relative of a proper perspective-projection camera model), are direct applications of Sections 4.2–4.3.
- **Word/token embeddings**: rotating an embedding space (e.g. via a learned or fitted orthogonal transform) to align two models' embedding spaces for comparison is a change-of-basis operation.

## 4.7 Worked example

Old basis: standard $\{e_1, e_2\}$. New basis: $b_1 = (1,1)$, $b_2=(-1,1)$ (a 45° rotation, scaled by $\sqrt2$).

$$P = \begin{bmatrix}1 & -1\\ 1 & 1\end{bmatrix}, \qquad P^{-1} = \frac{1}{2}\begin{bmatrix}1&1\\-1&1\end{bmatrix}$$

The vector $x_{\text{old}} = (3,1)$ has new-basis coordinates $x_{\text{new}} = P^{-1}x_{\text{old}} = (2,-1)$ — check: $2b_1 + (-1)b_2 = 2(1,1) - (-1,1) = (3,1)$. ✓. Full numeric version, plus a visualization of the axes, in `code/04_basis_and_change_of_basis.py`.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 13 (*Change of basis*), MIT 18.06 Lecture 31 (*Change of basis, image compression*).
