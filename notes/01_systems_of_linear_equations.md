# 1. Systems of Linear Equations

## 1.1 Setup

A system of $m$ linear equations in $n$ unknowns

$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\ \vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
$$

is written compactly as $Ax = b$, where $A \in \mathbb{R}^{m\times n}$, $x\in\mathbb{R}^n$, $b\in\mathbb{R}^m$.

**Two views worth holding simultaneously:**

- **Row view.** Each equation is a hyperplane in $\mathbb{R}^n$; a solution is a point lying on all $m$ hyperplanes at once.
- **Column view.** $Ax = x_1 a_1 + x_2 a_2 + \cdots + x_n a_n$, where $a_j$ is the $j$-th column of $A$. A solution is a set of weights that combines the columns of $A$ into $b$. This is the view that generalizes — "does $b$ live in the span of the columns of $A$?" is the question that governs everything below, and it's the same question you'll ask in Chapter 3 (Vector Spaces).

## 1.2 Gaussian elimination

Gaussian elimination reduces $[A \mid b]$ to **row echelon form** using three elementary row operations:

1. Swap two rows.
2. Scale a row by a nonzero constant.
3. Add a multiple of one row to another.

**Why these are safe:** each operation corresponds to left-multiplying by an invertible matrix $E$ (a permutation matrix, a diagonal scaling matrix, or an identity plus one off-diagonal entry, respectively). If $Ax=b$ then $EAx = Eb$, and since $E$ is invertible, the converse also holds ($EAx=Eb \Rightarrow Ax = E^{-1}Eb = b$). So elementary row operations never change the solution set — they only change how it's written down.

Row echelon form has:
- All-zero rows at the bottom.
- The leading (leftmost nonzero) entry of each row strictly to the right of the leading entry of the row above ("staircase" pattern).

Back-substitution then solves from the bottom row up, since the bottom nonzero row involves the fewest variables.

## 1.3 Existence and uniqueness

Let $\mathrm{rank}(A)$ denote the number of pivot columns (equivalently, the dimension of the column space — see Chapter 3).

**Rouché–Capelli theorem.** $Ax=b$ has a solution if and only if $\mathrm{rank}(A) = \mathrm{rank}([A\mid b])$.

*Why:* $b$ is in the span of the columns of $A$ exactly when appending $b$ as a column doesn't increase the rank. If it did increase the rank, $b$ would introduce a new pivot — meaning $b$ points somewhere the columns of $A$ can't reach.

Given consistency:

- **Unique solution** iff $\mathrm{rank}(A) = n$ (full column rank — every column is a pivot column, no free variables).
- **Infinitely many solutions** iff $\mathrm{rank}(A) < n$ (at least one free variable).

**General solution structure.** If $x_p$ is any particular solution ($Ax_p = b$) and $x_h$ is any solution of the homogeneous system $Ax_h = 0$, then $x_p + x_h$ is also a solution, and every solution has this form:

$$x = x_p + x_h, \qquad x_h \in \mathrm{null}(A)$$

*Proof.* If $Ax_1 = b$ and $Ax_2 = b$, then $A(x_1 - x_2) = Ax_1 - Ax_2 = b - b = 0$, so $x_1 - x_2 \in \mathrm{null}(A)$. Conversely if $x_h \in \mathrm{null}(A)$ then $A(x_p + x_h) = Ax_p + Ax_h = b + 0 = b$. $\blacksquare$

This is the same "particular + homogeneous" decomposition you'll see again for linear ODEs and for the bias/variance-free part of a linear model — it's a general pattern for any linear operator equation, not just this one.

## 1.4 LU decomposition (why elimination is worth factoring)

If Gaussian elimination on $A$ needs no row swaps, the elimination steps can be recorded as $A = LU$: $U$ is the upper-triangular row echelon form, and $L$ is unit lower-triangular, recording the multipliers used to zero out each entry below the pivots. (With row swaps, this becomes $PA = LU$ for a permutation matrix $P$.)

The payoff: solving $Ax=b$ for **many different $b$'s** with the same $A$ (e.g., re-solving a system at every timestep of a simulation) only requires elimination once. Each new $b$ is solved in $O(n^2)$ via two triangular solves — forward-substitute $Ly=b$, then back-substitute $Ux=y$ — instead of repeating the full $O(n^3)$ elimination.

## 1.5 Where this shows up in AI/ML

- **Linear regression's normal equations**, $A^\top A\, w = A^\top y$, are a system of linear equations in the weights $w$ — everything in this chapter (rank, consistency, when the solution is unique) directly governs whether a regression problem is well-posed. If $A^\top A$ is singular, the "unique solution" case fails and you're in **underdetermined** territory (more parameters than independent constraints) — exactly the regime deep nets live in, which is why regularization (ridge, weight decay) is really "restoring full rank."
- **Backpropagation-adjacent solves**: some second-order optimizers (Newton's method, natural gradient) solve a linear system $Hx = -g$ at every step, where $H$ is the Hessian or an approximation to it.
- **Physics-informed / simulation ML**: PDE discretizations reduce to large sparse linear systems solved with elimination-descended methods (though in practice with iterative solvers, not dense Gaussian elimination, once $n$ is large — see conjugate gradient / GMRES, out of scope here).
- **Constraint satisfaction in graphics/robotics**: inverse kinematics and mesh-deformation problems are frequently linear (or linearized) systems.

## 1.6 Worked example

$$
A = \begin{bmatrix} 2 & 1 \\ 4 & -1 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 1 \end{bmatrix}
$$

Augmented matrix and elimination ($R_2 \leftarrow R_2 - 2R_1$):

$$
\left[\begin{array}{cc|c} 2 & 1 & 5 \\ 4 & -1 & 1 \end{array}\right]
\longrightarrow
\left[\begin{array}{cc|c} 2 & 1 & 5 \\ 0 & -3 & -9 \end{array}\right]
$$

Back-substitute: $-3x_2 = -9 \Rightarrow x_2 = 3$; then $2x_1 + 3 = 5 \Rightarrow x_1 = 1$. Since both columns became pivot columns ($\mathrm{rank}(A) = 2 = n$), the solution $x = (1, 3)$ is unique — confirm with `code/01_systems_of_linear_equations.py`.

## 1.7 Common pitfalls

- **Conflating "no solution" with "the matrix is singular."** A singular (or non-square) $A$ can still have *infinitely many* solutions if $b$ happens to lie in the column space — singularity alone doesn't tell you existence, only that uniqueness is off the table if a solution exists.
- **Assuming more equations means a more constrained (or more solvable) system.** More rows than columns ($m>n$) generically makes the system *harder* to satisfy exactly (overdetermined — this is precisely the least-squares setting in Chapter 6), not easier.

## Further resources

See [`LEARNING_PATH.md`](../LEARNING_PATH.md) — for this topic specifically: 3Blue1Brown Ch. 1–3, MIT 18.06 Lectures 1–3.
