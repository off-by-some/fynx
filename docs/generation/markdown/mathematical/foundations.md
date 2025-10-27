# Mathematical Foundation of Adaptive Materialization

## 1. The Computation Graph Model

### 1.1 Basic Definitions

A reactive computation system is a directed acyclic graph (DAG) $\mathcal{G} = (N, D)$ where:

- $N$ is the set of **nodes** (observables and computations)
- $D \subseteq N \times N$ is the set of **dependencies**

Each node $n \in N$ has:
- A **value** $\text{val}(n) \in \mathcal{V}$ (some value domain)
- A **computation function** $f_n: \mathcal{V}^k \to \mathcal{V}$ where $k = |\text{parents}(n)|$
- A **state** $\sigma(n) \in \{\mathtt{Virtual}, \mathtt{Tracked}, \mathtt{Materialized}\}$

**Parent set:**
$$\text{parents}(n) = \{m \in N : (m, n) \in D\}$$

**Child set:**
$$\text{children}(n) = \{m \in N : (n, m) \in D\}$$

**Value computation:**
$$\text{val}(n) = \begin{cases}
\text{stored value} & \text{if } n \text{ is source observable} \\
f_n(\text{val}(p_1), \ldots, \text{val}(p_k)) & \text{if } n \text{ is computed, } \text{parents}(n) = \{p_1, \ldots, p_k\}
\end{cases}$$

### 1.2 Computation Chains

A **computation chain** from source $s$ to node $n$ is a path:
$$\pi = s \to n_1 \to n_2 \to \cdots \to n$$

The **composed function** along this chain is:
$$F_\pi = f_n \circ f_{n_{k-1}} \circ \cdots \circ f_{n_1}$$

So: $\text{val}(n) = F_\pi(\text{val}(s))$

**Key insight:** If we never store intermediate values, we compute $F_\pi$ directly. If we materialize at node $n_i$, we break the chain:
$$\text{val}(n) = f_n(\ldots f_{n_i+1}(\text{val}(n_i))\ldots)$$

where $\text{val}(n_i)$ is cached.

## 2. The Three States and Their Semantics

### 2.1 State Definitions

**Virtual State** ($\mathtt{Virtual}$):
$$\sigma(n) = \mathtt{Virtual} \iff \begin{cases}
\text{val}(n) \text{ is recomputed on every access} \\
\text{No entry in delta store} \\
\text{No change tracking}
\end{cases}$$

**Tracked State** ($\mathtt{Tracked}$):
$$\sigma(n) = \mathtt{Tracked} \iff \begin{cases}
\text{val}(n) \text{ stored in DeltaKVStore} \\
\text{Changes propagate via delta system} \\
\text{Dependencies registered}
\end{cases}$$

**Materialized State** ($\mathtt{Materialized}$):
$$\sigma(n) = \mathtt{Materialized} \iff \begin{cases}
\sigma(n) = \mathtt{Tracked} \text{ AND} \\
\text{Intermediate result cached for reuse}
\end{cases}$$

Note: In this system, $\mathtt{Materialized} \subset \mathtt{Tracked}$ (materialized nodes are tracked).

### 2.2 Access Cost Analysis

For a node $n$ at depth $d$ in a chain:

**Virtual access cost:**
$$C_{\text{access}}^{\mathtt{Virtual}}(n) = \sum_{i=1}^{d} c_i$$

where $c_i$ is the cost of computing $f_i$ at depth $i$.

**Tracked/Materialized access cost:**
$$C_{\text{access}}^{\mathtt{Tracked}}(n) = O(1)$$

### 2.3 Update Propagation Cost

When source $s$ updates, the **affected set** is:
$$\text{Affected}(s) = \{n \in N : \exists \text{ directed path from } s \text{ to } n\}$$

For virtual nodes: No propagation overhead (recomputed on demand)
$$C_{\text{propagate}}^{\mathtt{Virtual}} = 0$$

For tracked nodes: Must update all affected descendants
$$C_{\text{propagate}}^{\mathtt{Tracked}}(s) = \sum_{n \in \text{Affected}(s) \cap \mathtt{Tracked}} c_n$$

## 3. The Materialization Decision Problem

### 3.1 Optimization Objective

Given access frequency $\alpha_n$ and update frequency $\beta_n$ for node $n$, minimize:

$$\text{Cost}(n, \sigma) = \alpha_n \cdot C_{\text{access}}(n, \sigma) + \beta_n \cdot C_{\text{update}}(n, \sigma) + M(n, \sigma)$$

where $M(n, \sigma)$ is memory cost.

### 3.2 Fan-out Analysis

Define the **fan-out** of node $n$:
$$\phi(n) = |\text{children}(n)|$$

**Theorem (Fan-out Materialization):** If $\phi(n) \geq 2$ and children access $n$ with frequency $\alpha_1, \alpha_2, \ldots, \alpha_k$, then materializing $n$ reduces total cost.

**Proof:**

Without materialization, each child recomputes independently:
$$C_{\text{total}}^{\text{no-mat}} = \sum_{i=1}^{k} \alpha_i \cdot d \cdot \bar{c}$$

where $d$ is chain depth and $\bar{c}$ is average computation cost.

With materialization at $n$:
$$C_{\text{total}}^{\text{mat}} = \beta_n \cdot c_n + \sum_{i=1}^{k} \alpha_i \cdot (d - d_n) \cdot \bar{c}$$

where $d_n$ is depth to $n$.

The savings are:
$$\Delta C = \sum_{i=1}^{k} \alpha_i \cdot d_n \cdot \bar{c} - \beta_n \cdot c_n$$

Since typically $\sum \alpha_i \gg \beta_n$ (read-heavy workload), we get $\Delta C > 0$ for $k \geq 2$.

**Corollary:** The threshold $\tau = 2$ is optimal for the transition Virtual → Tracked/Materialized.

## 4. Function Fusion Mathematics

### 4.1 Composition vs Materialization

Consider a linear chain: $n_0 \xrightarrow{f_1} n_1 \xrightarrow{f_2} n_2 \xrightarrow{f_3} n_3$

**Strategy 1: Full Materialization**
- Store values at each $n_i$
- Access: $\text{val}(n_3) = \text{cache}[n_3]$ — $O(1)$
- Space: $O(4)$ — store all intermediate values

**Strategy 2: Function Fusion (Virtual)**
- Compose: $F = f_3 \circ f_2 \circ f_1$
- Access: $\text{val}(n_3) = F(\text{val}(n_0))$ — $O(c_1 + c_2 + c_3)$
- Space: $O(1)$ — store only $F$ and $n_0$

**Strategy 3: Adaptive (This System)**
- If $\phi(n_1) = 1$ and $\phi(n_2) = 1$: Use fusion (Strategy 2)
- If $\phi(n_i) \geq 2$ for any $i$: Materialize at $n_i$ (hybrid)

### 4.2 The Fusion Algebra

When building a chain in virtual mode, we compose functions:

$$\text{chain}_0 = \langle f_1, [s] \rangle$$
$$\text{chain}_1 = \langle f_2 \circ f_1, [s] \rangle$$
$$\text{chain}_2 = \langle f_3 \circ f_2 \circ f_1, [s] \rangle$$

Each chain is represented as $\langle F, \text{sources} \rangle$ where:
- $F$ is the composed function
- sources is the list of root observables

**SmartComputed.then() implementation:**

```
then(new_fn):
  if materialized OR fanout(self) ≥ 2:
    # Break chain, materialize this node
    mat = materialize(self)
    return ⟨new_fn, [mat]⟩
  else:
    # Continue fusion
    return ⟨new_fn ∘ self.fused_fn, self.sources⟩
```

**Theorem (Fusion Optimality):** For a linear chain with $\phi(n_i) = 1$ for all intermediate $n_i$, function fusion minimizes space while maintaining $O(d)$ access cost, which is unavoidable without caching.

## 5. StreamMerge: The Tuple Caching Strategy

### 5.1 Problem Formulation

Given $n$ observables $\{o_1, o_2, \ldots, o_n\}$, produce:
$$t = (o_1.\text{val}, o_2.\text{val}, \ldots, o_n.\text{val})$$

Updates occur when any $o_i$ changes: $o_i.\text{val} \to o_i.\text{val}'$

### 5.2 Naive Approach

Recompute entire tuple on each access:
$$\text{val}(t) = (\text{val}(o_1), \text{val}(o_2), \ldots, \text{val}(o_n))$$

Cost per access: $O(n)$

### 5.3 Cached Incremental Update

**Data structure:**
```
cached_values: Array[n]     // Individual values
cached_tuple: Tuple         // Combined tuple
```

**Initialization:**
$$\text{cached\_values}[i] = o_i.\text{val} \quad \forall i \in [1, n]$$
$$\text{cached\_tuple} = \text{tuple}(\text{cached\_values})$$

**Update protocol:**

When $o_j$ changes from $v$ to $v'$:

1. **Partial update:** $\text{cached\_values}[j] \gets v'$ — $O(1)$
2. **Tuple reconstruction:** $\text{cached\_tuple} \gets \text{tuple}(\text{cached\_values})$ — $O(n)$
3. **Notify subscribers:** broadcast $\text{cached\_tuple}$ — $O(k)$ for $k$ subscribers

**Access:** Return $\text{cached\_tuple}$ — $O(1)$

### 5.4 Cost Analysis

**Amortized update cost:**

For $m$ updates across $n$ sources:
$$C_{\text{avg}} = \frac{m \cdot O(n)}{m} = O(n)$$

But this is optimal! We must reconstruct the tuple (immutable in Python), requiring $O(n)$ to allocate.

**Memory efficiency:**

- Naive: $O(k \cdot n)$ for $k$ subscribers (each caches their own tuple)
- Cached: $O(n)$ (single shared tuple)

**Savings factor:** $k$ (significant for many subscribers)

## 6. Delta-Based Change Propagation

### 6.1 Topological Propagation Order

For a DAG $\mathcal{G} = (N, D)$, there exists a **topological ordering** $\prec$ such that:
$$\forall (u, v) \in D: u \prec v$$

When source $s$ updates, we propagate changes in topological order:

**Algorithm:**
```
UpdatePropagate(s, new_value):
  1. Set val(s) ← new_value
  2. Compute Affected(s) = {n : s leads to n}
  3. Sort Affected(s) by topological order
  4. For each n ∈ Affected(s) in order:
       if σ(n) = Tracked:
         val(n) ← f_n(val(parents(n)))
         emit_delta(n, val(n))
```

### 6.2 Complexity Analysis

**Topological sort:** $O(|N| + |D|)$ (one-time cost, cached)

**Per-update propagation:**
$$C_{\text{propagate}} = O(|\text{Affected}(s)|) + \sum_{n \in \text{Affected}(s)} O(c_n)$$

This is **optimal** because:
1. We only visit affected nodes
2. Each node computed exactly once (topological order ensures parents ready)
3. No redundant computation

### 6.3 Why Delta-Based Beats Full Reevaluation

**Full reevaluation:**
$$C_{\text{full}} = O(|N|) + \sum_{n \in N} O(c_n)$$

Must check all nodes even if unchanged.

**Delta-based:**
$$C_{\text{delta}} = O(|\text{Affected}(s)|) + \sum_{n \in \text{Affected}(s)} O(c_n)$$

**Speedup ratio:**
$$\rho = \frac{C_{\text{full}}}{C_{\text{delta}}} = \frac{|N|}{|\text{Affected}(s)|}$$

In typical applications:
- $|N| \sim 100\text{-}1000$ nodes
- $|\text{Affected}(s)| \sim 5\text{-}20$ nodes
- $\rho \sim 10\text{-}100\times$ speedup

## 7. The Complete State Transition System

### 7.1 State Machine for Observables

Each observable node transitions between states:

```
       subscribe()              fanout ≥ 2
Virtual --------→ Tracked ←------------------- SmartComputed
                     ↓
                     ↓ register_dependent() × 2
                     ↓
                 Materialized
```

**Transition rules:**

$$\sigma(n)_{t+1} = \begin{cases}
\mathtt{Tracked} & \text{if } \sigma(n)_t = \mathtt{Virtual} \land (\text{subscribed}(n) \lor \phi(n) \geq 2) \\
\mathtt{Materialized} & \text{if } \sigma(n)_t = \mathtt{Tracked} \land \phi(n) \geq 2 \land \text{is\_computed}(n) \\
\sigma(n)_t & \text{otherwise}
\end{cases}$$

### 7.2 SmartComputed State Transitions

For computed nodes (SmartComputed):

**State invariants:**
- Virtual: $\phi(n) < 2$ (linear chain)
- Materialized: $\phi(n) \geq 2$ (branch point)

**Chain building:**

Building $n_1 >> n_2 >> n_3$:

- $n_1$: Virtual, $\phi(n_1) = 1$, fused function $= f_1$
- $n_2$: Virtual, $\phi(n_1)$ still $= 1$, fused function $= f_2 \circ f_1$
- $n_3$: Virtual, $\phi(n_1)$ still $= 1$, fused function $= f_3 \circ f_2 \circ f_1$

**Branching:**

Now create $n_4 = n_2 >> f_4$:

- $\phi(n_2) \gets 2$ (two dependents: $n_3$ and $n_4$)
- Trigger materialization of $n_2$
- Break fusion: $n_2$ now stores $f_2 \circ f_1$ result
- $n_3$ becomes: $f_3(n_2.\text{val})$ — reads from cache
- $n_4$ becomes: $f_4(n_2.\text{val})$ — reads from cache

## 8. Conditional Observables: Emission Filtering

### 8.1 Problem Statement

Given source $s$ and condition $c$ (boolean observable), emit only when:
1. Condition transitions false → true
2. Value changes while condition is true

**State space:**
- $c$: condition (true/false)
- $v$: value
- $\text{active}$: was condition true last time?

### 8.2 Emission Logic

**State transitions:**

$$\text{emit}(v, c, \text{active}_{\text{prev}}) = \begin{cases}
\text{true} & c = \text{true} \land \text{active}_{\text{prev}} = \text{false} & \text{(activation)} \\
\text{true} & c = \text{true} \land \text{active}_{\text{prev}} = \text{true} \land v \neq v_{\text{prev}} & \text{(value change)} \\
\text{false} & \text{otherwise}
\end{cases}$$

**Theorem (Emission Reduction):** For $u$ total updates with condition true probability $p$ and value change probability $q$:

Expected emissions:
$$\mathbb{E}[\text{emissions}] \leq pu + pqu$$

Naive approach emits all $u$ updates.

Reduction factor:
$$\frac{u}{pu + pqu} = \frac{1}{p(1 + q)}$$

For sparse conditions ($p \ll 1$), this gives $O(1/p)$ reduction.

## 9. System-Wide Optimization: The Complete Picture

### 9.1 Hybrid Strategy

The system uses a **three-tier hybrid**:

1. **Virtual tier:** Linear chains with no branching
   - No memory overhead
   - Function composition
   - Lazy evaluation

2. **Tracked tier:** Subscribed or fan-out nodes
   - Change propagation
   - Delta-based updates
   - O(affected) complexity

3. **Materialized tier:** Branch points with $\phi \geq 2$
   - Cached intermediate results
   - Eliminates redundant computation
   - Shared across dependents

### 9.2 Total System Cost

For a graph with:
- $N_v$ virtual nodes
- $N_t$ tracked nodes
- $N_m$ materialized nodes

**Memory:**
$$M = O(N_v) + O(N_t + E_t) + O(N_m + E_m + \sum_{n \in N_m} |v_n|)$$

where $E_t$, $E_m$ are edges for tracked/materialized nodes.

**Update cost:**
$$C_{\text{update}} = \sum_{n \in \text{Affected} \cap (N_t \cup N_m)} c_n$$

**Access cost:**
$$C_{\text{access}}(n) = \begin{cases}
O(d_n) & n \in N_v \\
O(1) & n \in N_t \cup N_m
\end{cases}$$

### 9.3 Why This Is Optimal

**Theorem (Optimality):** The adaptive strategy minimizes:
$$\sum_{n \in N} \alpha_n C_{\text{access}}(n) + \beta_n C_{\text{update}}(n) + M(n)$$

subject to constraints:
1. $C_{\text{update}}(n) \geq O(|\text{Affected}(n)|)$ (must visit affected)
2. $C_{\text{access}}(n) = O(1) \Rightarrow M(n) \geq \Omega(1)$ (caching requires memory)

**Proof sketch:**
- Virtual nodes: $\alpha_n$ low, no caching needed → minimize $M(n)$
- Tracked nodes: $\alpha_n$ high or $\beta_n$ high → cache for $O(1)$ access
- Fan-out: $\phi(n) \geq 2$ → materialize to prevent $\phi(n) \times$ recomputation
- Each decision minimizes local cost; no global conflict

## 10. Summary: The Mathematical Beauty

The system achieves optimality through:

1. **Adaptive state transitions** based on observed usage patterns
2. **Function composition** for memory efficiency in linear chains
3. **Strategic materialization** at branch points for computation sharing
4. **Delta-based propagation** for $O(\text{affected})$ update complexity
5. **Lazy evaluation** to defer costs until necessary
6. **Incremental caching** in StreamMerge for $O(1)$ tuple access

The threshold $\tau = 2$ emerges naturally from the break-even analysis, and the three-tier system (Virtual/Tracked/Materialized) provides optimal tradeoffs across the memory-computation-latency space.
