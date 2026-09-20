# BharatOpt Mathematical Formulation & Architectural Manual

**BharatOpt** is an indigenous, sovereign GPU-accelerated optimization engine engineered specifically for refinery flowsheet optimization, multi-period crude scheduling, and non-linear blend pooling (SIH26119 • Team NovaKin • MRPL - Mangalore Refinery and Petrochemicals Limited).

---

## 1. Linear Programming & PDHG First-Order Solver

BharatOpt solves large-scale linear programs in standard primal-dual form:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x + c_0 \\
\text{s.t.} \quad & b^L \le A x \le b^U \\
& l \le x \le u
\end{aligned}
$$

### 1.1 Primal-Dual Hybrid Gradient (Chambolle-Pock)
To scale across millions of non-zero coefficients on multi-threaded CPUs and CUDA streaming multiprocessors without memory bottlenecks of matrix factorization, BharatOpt applies the first-order **Primal-Dual Hybrid Gradient (PDHG)** algorithm:

1. **Dual Update**:
   $$y^{k+1} = \text{prox}_{\sigma g^*}\left(y^k + \sigma A \bar{x}^k\right)$$
   where the proximal projection enforces row constraints $b^L \le A x \le b^U$.

2. **Primal Extrapolation**:
   $$x^{k+1} = \Pi_{[l, u]}\left(x^k - \tau (c + A^T y^{k+1})\right)$$

3. **Momentum Acceleration**:
   $$\bar{x}^{k+1} = x^{k+1} + \theta (x^{k+1} - x^k), \quad \theta \in [0, 1]$$

Convergence is unconditionally guaranteed when step sizes satisfy:
$$\tau \sigma \|A\|_2^2 < 1$$
BharatOpt employs power-iteration to compute the operator norm $L = \|A\|_2$ and dynamically balances primal and dual residuals via adaptive Barzilai-Borwein step scaling every 50 iterations.

---

## 2. Bilinear Pooling Problem Formulation

Refinery blending operations exhibit severe non-convex bilinearities because properties (such as Research Octane Number (RON), sulfur wt%, Reid Vapor Pressure (RVP), and density) blend in intermediate pools before product delivery.

### 2.1 The P-Formulation
Let $I$ be the set of feed streams, $L$ be the set of intermediate pools, and $J$ be the set of final products:
- $v_{il}$: volume flow from source $i$ to pool $l$.
- $v_{lj}$: volume flow from pool $l$ to product $j$.
- $C_{ik}$: concentration of quality $k$ in stream $i$.
- $p_{lk}$: concentration of quality $k$ in pool $l$ (unknown variable).

**Pool mass conservation**:
$$\sum_{i \in I} v_{il} = \sum_{j \in J} v_{lj}, \quad \forall l \in L$$

**Pool quality balance (Bilinear)**:
$$\sum_{i \in I} C_{ik} v_{il} = p_{lk} \sum_{j \in J} v_{lj}, \quad \forall l \in L, \forall k$$

The non-linearity arises from the product of pool quality $p_{lk}$ and discharge rate $v_{lj}$.

---

## 3. Global Optimization via Spatial Branch-and-Bound

To guarantee global optimality rather than getting trapped in local sub-optima, BharatOpt deploys **Spatial Branch-and-Bound (sB&B)** using convex McCormick relaxations:

### 3.1 McCormick Envelopes for Bilinear Terms $w = x \cdot y$
Given bounds $x \in [x^L, x^U]$ and $y \in [y^L, y^U]$:

$$
\begin{aligned}
w &\ge x^L y + x y^L - x^L y^L \\
w &\ge x^U y + x y^U - x^U y^U \\
w &\le x^U y + x y^L - x^U y^L \\
w &\le x^L y + x y^U - x^L y^U
\end{aligned}
$$

### 3.2 Best-Bound Search Strategy
BharatOpt manages open search nodes with an explicit max-priority heap ordered by the relaxation's certified bound. At each node:
1. Implied variable bounds are tightened via singleton scanning.
2. The convex McCormick LP relaxation is solved.
3. If the relaxation is infeasible or bounded worse than the incumbent best feasible solution, the node is pruned.
4. Otherwise, the bilinear term with maximum violation $|w^* - x^* y^*|$ is branched by bisecting the domain.

---

## 4. Infeasibility Certificates & Bounded IIS

When production constraints or crude deliveries cannot satisfy product specifications, BharatOpt provides:
1. **Farkas Multipliers**: Dual rays $y \ge 0$ satisfying $A^T y \le 0$ and $b^T y > 0$, certifying that no primal feasible solution exists.
2. **Bounded Irreducible Infeasible Subsystems (IIS)**: Identifies the minimal conflicting subset of refinery constraints (e.g. conflicting Euro-VI 10 ppm sulfur limits vs available crude sweetening capacity) within 2.5 seconds using deletion filtering with warm starts.

---

## 5. Scope-1 & Scope-3 Carbon Footprint Integration

In addition to financial Gross Refining Margin (GRM), BharatOpt directly models environmental externalities:
- **Scope 1 Direct Process Emissions**: $E_1 = \sum_{u \in \text{Units}} e_u \cdot \text{Throughput}_u$
- **Scope 3 Upstream Feedstock Footprint**: $E_3 = \sum_{f \in \text{Crudes}} c_f \cdot \text{Procurement}_f$
- **Carbon Tax Penalty**: Adds $\tau_{\text{carbon}} \cdot (E_1 + E_3)$ to the planning objective function.
