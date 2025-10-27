# FynX: Adaptive Materialization in Reactive Systems

> Reactive graphs are predominantly linear, with branch points appearing infrequently. The resulting patterns—linear chains, branch points, and subscriptions—provide sufficient information for automatic, intelligent caching decisions.


When you connect values through computations, you create a web of dependencies where changes ripple outward through dependent calculations. The fundamental question becomes: which intermediate results should you cache? Cache everything and you waste memory on values computed once. Cache nothing and you recompute the same values repeatedly. Traditional reactive systems force you to make these decisions explicitly, annotating nodes as "hot" or "cold," "lazy" or "eager."

```javascript
// You must decide:
const hot = signal(x);     // cache
const cold = computed(() => x * 2); // don't cache
```

FynX takes a different approach. It watches how you build your computation graph and uses structural patterns—particularly how many consumers depend on each computation—to automatically decide when caching helps. The system starts entirely virtual, creating no intermediate storage. As your graph takes shape and branches appear, materialization happens exactly where sharing occurs.

```python
y = x >> (lambda v: v * 2)  # System: "One consumer? → virtual."
# Later on in the code...
w = y >> f
z = y >> g  # "Two consumers? → materialize y."
```

## How Computation Graphs Work

Think of observables as values that can change over time. Each observable either holds a source value directly, or computes its value from other observables. When you write `y = x >> (lambda v: v * 2)`, you're saying "y is whatever x is, doubled." The system represents this as a node that knows how to compute its value by asking x for its value and applying the function.

Formally, we can model this as a directed acyclic graph (DAG) where nodes are observables and edges are dependencies. Each node $n$ has a value $\text{val}(n)$ and optionally a computation function $f_n$:

$$
\text{val}(n) = \begin{cases}
\text{stored value} & \text{if } n \text{ is a source} \\
f_n(\text{val}(p_1), \dots, \text{val}(p_k)) & \text{if } n \text{ has parents } p_1, \dots, p_k
\end{cases}
$$

A computation graph emerges naturally from these dependencies. Sources sit at the leaves—they store actual values. Computed nodes sit higher up, each knowing which nodes it depends on and how to combine their values. When a source changes, the system needs to figure out which computed values might have changed and let them know.

The interesting part is how values get stored and recomputed. A naive implementation would cache every intermediate result, but most computation chains are linear: `x >> f1 >> f2 >> f3` creates a straight line where each node has exactly one consumer. Caching these intermediate values wastes memory since they're only used once. A smarter approach keeps the entire chain virtual—just composed functions—and computes the final result directly from the source.

## The Three States

Every observable in FynX exists in one of three states, each representing a different trade-off between memory and computation.

### Virtual: Pure Computation

Virtual nodes are pure function closures with no storage overhead. When you access their value, they compute it fresh by asking their dependencies and applying the computation function:

```python
x = observable(5)                    # Virtual source
y = x >> (lambda v: v * 2)          # Virtual: just a function closure
z = y >> (lambda v: v + 3)          # Virtual: fused with y's function

print(z.value)  # Computes: (5 * 2) + 3 = 13
# No intermediate storage for y or z
# State: x=[virtual], y=[virtual], z=[virtual]
```

This looks inefficient—recomputing on every access—but it's optimal for linear chains. The >> operator fuses functions together, so `z` directly computes `(x * 2) + 3` rather than creating two separate nodes. You get the expressiveness of step-by-step transforms with the performance of a single composed function.

The cost of accessing a virtual node at depth $d$ is:

$$
C_{\text{read}}^{\text{virtual}} = \sum_{i=1}^{d} c_i
$$

where $c_i$ is the computation cost at each level. For a fused chain, this collapses to a single composed function evaluation, making it $O(d)$ but with minimal constant factors.

### Tracked: Change Propagation

Tracked nodes register with the key-value store and participate in reactive updates. The transition happens automatically when the system needs to propagate changes:

```python
x = observable(5)                    # Virtual source
y = x >> (lambda v: v * 2)          # Virtual: fused function
z = y >> (lambda v: v + 3)          # Virtual: fused function

# Subscribing forces tracking so changes can propagate
z.subscribe(lambda val: print(f"z changed to {val}"))

# Now: x=[tracked], y=[virtual fused], z=[materialized]
# The subscription materialized z, which triggered tracking x
# But y stayed virtual because it has only one consumer (z)

x.set(7)  # Prints: "z changed to 17"
```

Tracking enables the dependency graph to route updates correctly. The store maintains two mappings:
- $\mathcal{K}$: keys to current values
- $\mathcal{D}$: keys to sets of dependent keys

When `x` changes, the system uses $\mathcal{D}$ to find all nodes that transitively depend on `x`. For tracked nodes, access cost drops to constant time:

$$
C_{\text{read}}^{\text{tracked}} = O(1)
$$

This is a hash table lookup rather than recomputation. But notice that `y` stays virtual—it's still just a fused function inside `z`'s computation.

### Materialized: Shared Results

Materialized nodes cache their computed results. Materialization happens automatically when a computation has multiple consumers—the moment when caching pays for itself:

```python
x = observable(5)                    # Virtual source
y = x >> (lambda v: v * 2)          # Virtual: single consumer
z = y >> (lambda v: v + 3)          # Virtual: y still has one consumer

# Now create a branch - y gains a second consumer
w = y >> (lambda v: v - 1)
# State: x=[tracked], y=[materialized], z=[virtual], w=[virtual]

print(y.value)  # Fast: O(1) lookup returns 10
print(z.value)  # Computes: 10 + 3 = 13
print(w.value)  # Computes: 10 - 1 = 9
```

The instant `w` is created, `y` materializes because it now has two consumers (`z` and `w`). The system tracks `x` so it can invalidate `y` when `x` changes. But `z` and `w` remain virtual—they're pure functions that compute from `y`'s cached value.

The cost model makes the threshold of 2 dependents obvious. Let $\phi(n)$ be the fan-out (number of consumers) of node $n$, and let $c_n$ be its computation cost. Without materialization, total cost for all consumers is:

$$
C_{\text{total}}^{\text{virtual}} = \phi(n) \times c_n
$$

With materialization, you compute once and look up $\phi(n)$ times:

$$
C_{\text{total}}^{\text{materialized}} = c_n + \phi(n) \times O(1)
$$

Materialization wins when $\phi(n) \geq 2$, which is exactly the threshold FynX uses.

The transition between states is one-way and automatic. Virtual nodes become tracked when they need to participate in change propagation. Tracked nodes become materialized when fan-out appears. You never specify these states manually.

## When Materialization Happens

The key insight is that fan-out indicates sharing. When a computation has two or more consumers, you'll access that value multiple times, making caching worthwhile.

### Example 1: Linear Chain Stays Virtual

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)
w = z >> (lambda v: v / 2)

# State: All virtual, fully fused
# w computes directly as: (((5 * 2) + 3) / 2)
# No intermediate storage
```

Every node has exactly one consumer (or zero), so everything stays virtual. The >> operator fuses the functions into a single composition. Accessing `w.value` computes the entire chain in one go.

### Example 2: Branch Point Triggers Materialization

```python
x = observable(5)
y = x >> (lambda v: v * 2)
# State: x=[virtual], y=[virtual]

z = y >> (lambda v: v + 3)
# State: x=[virtual], y=[virtual], z=[virtual]
# y still has only one consumer

w = y >> (lambda v: v - 1)
# NOW y has two consumers!
# State: x=[tracked], y=[materialized], z=[virtual], w=[virtual]
```

The fourth line changes everything. Before it, `y` was virtual with one consumer. After it, `y` has two consumers and materializes. The system also tracks `x` so changes can propagate to `y`. But `z` and `w` remain virtual—they compute directly from `y`'s cached value.

### Example 3: Subscription Forces Materialization

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)
# State: All virtual (fused chain)

z.subscribe(lambda val: print(f"z = {val}"))
# State: x=[tracked], y=[virtual fused], z=[materialized]
```

Subscriptions need change propagation, which requires materialization. Now when `x` changes, the system knows to recompute and emit `z`'s new value to the callback. Interestingly, `y` stays virtual—it's still just part of `z`'s fused computation function.

### Example 4: Fusion Stops at Materialization

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)
# State: All virtual, z computes as (x * 2) + 3

w = y >> (lambda v: v - 1)
# State: x=[tracked], y=[materialized], z=[virtual], w=[virtual]
# y materialized due to branch

# Now add more transforms
a = z >> (lambda v: v * 10)
b = w >> (lambda v: v * 10)
# State: a=[virtual], b=[virtual]
# Both fuse with their respective parents (z and w)
```

Once `y` materializes at the branch point, new chains can extend from it but fusion happens independently in each branch. The `z >> transform` operations create new virtual nodes that fuse with `z`, and the `w >> transform` operations do likewise. Each branch maintains its own fusion chain starting from the materialized value `y`.

The threshold is exactly two consumers. Why two? The cost model makes it clear: without caching, you pay the computation cost k times for k consumers. With caching, you pay it once plus k cheap lookups. The break-even point is k=2.

Formally, for a node with computation cost $c$ and $k$ consumers:

$$\text{Without caching: } k \cdot c$$
$$\text{With caching: } c + k \cdot O(1)$$

Caching becomes beneficial when $k \cdot c > c + k \cdot O(1)$, which simplifies to $k > 1 + \frac{O(1)}{c - O(1)}$. For any non-trivial computation where $c \gg O(1)$, the threshold is effectively $k \geq 2$.

## How Changes Propagate

When you change a source value, the system needs to update affected computations. FynX uses a hybrid strategy: materialized nodes update eagerly, virtual nodes update lazily.

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)
w = y >> (lambda v: v - 1)
# State: x=[tracked], y=[materialized], z=[virtual], w=[virtual]

x.set(7)
# What happens:
# 1. Store updates x's value to 7
# 2. Dependency graph finds y depends on x
# 3. y recomputes: 7 * 2 = 14 (stored)
# 4. z and w are virtual, so they wait
# 5. Next access to z.value computes: 14 + 3 = 17
# 6. Next access to w.value computes: 14 - 1 = 13
```

The materialized node `y` updates immediately because it's cached in the store and the dependency graph knows about it. The virtual nodes `z` and `w` don't exist in the store—they're just functions—so they recompute lazily on the next access.

When a source $s$ changes, the system identifies all affected nodes $\text{Affected}(s)$ and updates them in topological order. For a node at depth $d$, the write cost is:

$C_{\text{write}}(s) = O(|\text{Affected}(s)|) + \sum_{n \in \text{Affected}(s)} c_n$

This is optimal—you can't do better than $\Omega(|\text{Affected}(s)|)$ since every affected node must be visited. The topological sort ensures correct update order: if $y$ depends on $x$ and $z$ depends on $y$, then $y$ updates before $z$.

### Propagation with Subscriptions

When observables have subscriptions, they're materialized and participate in eager propagation:

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)

z.subscribe(lambda val: print(f"z = {val}"))
# Prints: "z = 13"
# State: x=[tracked], y=[virtual fused], z=[materialized]

x.set(7)
# Prints: "z = 17"
# What happens:
# 1. Store updates x to 7
# 2. Dependency graph finds z depends on x
# 3. z recomputes: (7 * 2) + 3 = 17
# 4. z's subscription fires with new value
```

The subscription forced `z` to materialize, so it updates eagerly and notifies its callback. Notice that `y` stayed virtual—it's fused into `z`'s computation, so there's no separate node to update.

## Fusion: Collapsing Computation Chains

When you chain transforms with `>>`, FynX doesn't create separate nodes for each step. Instead, it fuses them into a single composed function. This is why linear chains stay efficient.

### How Fusion Works

```python
# Step by step construction
x = observable(5)
# State: x=[virtual]

y = x >> (lambda v: v * 2)
# State: x=[virtual], y=[virtual]
# y's function: λ(x_val) → x_val * 2

z = y >> (lambda v: v + 3)
# State: x=[virtual], y=[virtual], z=[virtual]
# z's function: λ(x_val) → (x_val * 2) + 3
# Note: z computes directly from x, skipping y!

w = z >> (lambda v: v / 2)
# State: x=[virtual], y=[virtual], z=[virtual], w=[virtual]
# w's function: λ(x_val) → ((x_val * 2) + 3) / 2
# The entire chain is a single composed function
```

Each time you write `observable >> function`, the system checks: does this observable have multiple dependents yet? If not, it creates a new virtual observable with a composed function that computes directly from the original source. The intermediate observables (`y` and `z`) still exist—you can access their values—but they don't create separate storage or computation nodes.

### When Fusion Stops

Fusion stops at materialization boundaries:

```python
x = observable(5)
y = x >> (lambda v: v * 2)
z = y >> (lambda v: v + 3)
# Fully fused: z computes as (x * 2) + 3

# Create branch - y materializes
w = y >> (lambda v: v - 1)
# State: x=[tracked], y=[materialized], z=[virtual], w=[virtual]

# Now extend z
a = z >> (lambda v: v * 10)
# State: a=[virtual]
# a's function: λ(y_val) → (y_val + 3) * 10
# a fuses with z but computes from y's cached value

# Extend w
b = w >> (lambda v: v * 10)
# State: b=[virtual]
# b's function: λ(y_val) → (y_val - 1) * 10
# b fuses with w, also computing from y's cached value
```

Once `y` materializes due to the branch, new chains extend from that materialized point. The `z` branch and `w` branch each maintain their own fusion chains, but both start from `y`'s cached value. This gives you the best of both worlds: shared caching where values split, and zero-overhead fusion in each linear continuation.

### Comparing With and Without Fusion

Without fusion, the naive approach creates separate nodes:

```python
# Naive (what you DON'T get):
x = observable(5)           # Node 1
y = x >> f1                 # Node 2: depends on node 1
z = y >> f2                 # Node 3: depends on node 2
w = z >> f3                 # Node 4: depends on node 3

# Accessing w.value:
# - Lookup node 4's value
# - Compute f3(node 3's value)
# - Need node 3's value, compute f2(node 2's value)
# - Need node 2's value, compute f1(node 1's value)
# Four lookups, three function calls
```

With fusion, you get a single computation:

```python
# FynX (what you DO get):
x = observable(5)           # Node 1
y = x >> f1                 # Virtual: fused
z = y >> f2                 # Virtual: fused
w = z >> f3                 # Virtual: fused

# Accessing w.value:
# - Compute (f3 ∘ f2 ∘ f1)(x.value)
# One lookup, one composed function call
```

The fused version uses constant memory regardless of chain length, while the naive version grows linearly. More importantly, the fused version has minimal allocation overhead—no intermediate values, no node objects, just function closures.

## Stream Merging and Tuples

A fundamental pattern in reactive systems is coordinating multiple independent observables. Consider a computation that depends on both a temperature sensor and a pressure reading, or a UI element that needs both user input and application state. The naive approach creates a computed observable that reads both sources:

```python
temp = observable(20)
pressure = observable(101.3)

# Naive: Create a new tuple on every access
combined = temp >> (lambda t: (t, pressure.value))
```

But this creates an asymmetry: `combined` formally depends only on `temp`, even though it reads `pressure`. Changes to `pressure` won't trigger updates to subscribers of `combined`. You need explicit dependencies on all sources.

FynX's stream merge operator solves this by treating tuple formation as a first-class operation:

```python
temp = observable(20)
pressure = observable(101.3)
humidity = observable(65)

# Create a proper multi-source dependency
sensors = temp + pressure + humidity
print(sensors.value)  # (20, 101.3, 65)

# Compute over all sources
comfort_index = sensors >> (lambda t, p, h: calculate_comfort(t, p, h))
```

The `+` operator creates a StreamMerge node that explicitly depends on all sources. Changes to any source—temperature, pressure, or humidity—will propagate through the dependency graph and trigger recomputation of `comfort_index`.

### Incremental Tuple Updates

The interesting implementation detail is how StreamMerge maintains its cached tuple. Rather than reconstructing the entire tuple on every source change, it updates only the changed position:

```python
sensors = temp + pressure + humidity
# Internal state: tuple_cache = (20, 101.3, 65)

temp.set(22)
# Update: tuple_cache[0] = 22
# Result: tuple_cache = (22, 101.3, 65)
# Cost: O(k) where k = number of changed sources
```

For a merge of $n$ sources, the tuple representation requires $\Theta(n)$ space. A naive reconstruction on each change would cost $\Theta(n)$ time to build the new tuple. The incremental approach reduces this to $O(k)$ where $k$ is the number of sources that changed in a single update cycle—typically $k = 1$.

This matters for wide merges. A dashboard might combine dozens of independent observables: user preferences, live data feeds, UI state, cached computations. Reconstructing a 50-element tuple on every change wastes work. Updating only changed positions keeps the cost proportional to actual changes.

### Lazy Subscription Semantics

StreamMerge uses lazy subscription to avoid cascading eager evaluation during graph construction:

```python
a = observable(1)
b = observable(2)
c = observable(3)

# Creating the merge doesn't subscribe to sources yet
merged = a + b + c
# State: merged caches (1, 2, 3) but has no active subscriptions

# Reading the value is fast - just return cached tuple
print(merged.value)  # O(1) lookup

# Subscribing triggers source subscriptions
merged.subscribe(lambda val: print(f"Update: {val}"))
# Now: merged subscribes to a, b, and c
# Changes will propagate reactively

a.set(10)  # Prints: "Update: (10, 2, 3)"
```

Without lazy subscription, creating a merge would immediately subscribe to all sources, which subscribe to their sources, cascading through the graph. In a large application with hundreds of observables, this eager subscription can dominate construction time. Lazy subscription defers this cost: sources are only subscribed when someone actually subscribes to the merge.

The cached tuple ensures that reading `merged.value` before any subscription still works efficiently. The tuple is maintained eagerly (updated on source changes if we're subscribed) or computed on-demand (first access if we're not subscribed). This gives you both fast reads and deferred subscription costs.

## Conditional Observables

Many reactive computations have natural activation conditions. A search results observable should only update when the search query is non-empty. A notification badge should only show when unread messages exist. An animation should only run when visible on screen.

The typical approach uses filtering at the subscription level—compute everything, then discard unwanted updates in the callback. But this wastes work: you're computing values you'll immediately throw away. What you want is conditional computation: only evaluate when the condition is satisfied.

```python
search_query = observable("")
api_results = search_query >> (lambda q: expensive_api_call(q))

# Problem: expensive_api_call runs even for empty queries!
# You'll filter in the subscription, but the work already happened
```

FynX's conditional observables solve this by incorporating the condition into the computation graph:

```python
search_query = observable("")
is_valid = search_query >> (lambda q: len(q) >= 3)

# Only compute when query is at least 3 characters
api_results = search_query & is_valid
```

The `&` operator creates a conditional observable that only evaluates its source when the condition is true. If you access `api_results.value` while `is_valid` is false and has never been true, FynX raises `ConditionNeverMet` to distinguish "never active" from "currently inactive."

### State Tracking and Caching

Conditional observables maintain three pieces of state:
- $\text{is\_active}$: whether the condition is currently satisfied
- $\text{has\_been\_active}$: whether the condition was ever satisfied
- $\text{last\_value}$: the most recent emitted value

This enables efficient caching behavior:

```python
x = observable(5)
condition = observable(False)
filtered = x & condition

# Initial state: never active
try:
    print(filtered.value)
except ConditionNeverMet:
    print("Never activated")  # Prints this

# Condition becomes true: transition to active
condition.set(True)
print(filtered.value)  # 5 (computed from source)
# State: active, has_been_active = True, last_value = 5

# Condition becomes false: cache last value
condition.set(False)
print(filtered.value)  # Still 5 (cached)
# State: inactive, but returns cached last_value

# Source changes while inactive
x.set(10)
print(filtered.value)  # Still 5 (doesn't update while inactive)

# Reactivation: compute fresh from source
condition.set(True)
print(filtered.value)  # 10 (recomputes from current source)
# State: active again, last_value = 10
```

The caching behavior is subtle but important. When the condition becomes false, the observable doesn't immediately "forget" its value—it caches `last_value` so repeated accesses during the inactive period are cheap. This matters for UI rendering where you might read the same observable many times per frame.

But crucially, source changes don't propagate while inactive. If `x` changes from 5 to 10 while the condition is false, `filtered` doesn't recompute or update its cache. This prevents wasted computation: why recalculate a value you won't use? Only when the condition becomes true again does the observable recompute from the current source state.

### Composing Conditionals

Conditionals compose naturally with other operators:

```python
search_query = observable("")
is_valid = search_query >> (lambda q: len(q) >= 3)
api_results = (search_query & is_valid) >> expensive_api_call

# Only calls expensive_api_call when query is valid
# Multiple transformations after the condition
processed = (search_query & is_valid) >> api_call >> parse_results >> sort

# Chained conditions: both must be true
is_authed = observable(True)
filtered = (search_query & is_valid) & is_authed
```

Each conditional in a chain acts as a gate, allowing values through only when its condition is satisfied. This creates a natural flow control mechanism in the computation graph without manual conditional logic scattered through callbacks.

## Why This Works

The adaptive approach succeeds because it respects the actual structure of computation graphs in reactive applications. Most dependency chains are linear—you transform a value through several steps but only use the final result. Creating intermediate storage for each step wastes memory without any access time benefit.

Branch points are relatively rare but incredibly important. When they appear, they signal genuine sharing: multiple parts of your application depend on the same computed value. This is exactly when caching pays for itself—computing once and looking up many times is cheaper than computing many times.

The threshold of two dependents emerges naturally from the cost model. Below two, the overhead of tracking and storing outweighs the benefit. At two or more, sharing dominates and caching wins. Using this threshold everywhere gives good performance without manual tuning.

Fusion amplifies these benefits by keeping chains maximally virtual. Even relatively long chains stay as single composed functions until a branch forces materialization. This means the system scales gracefully: adding more linear transforms doesn't create more nodes, it just composes deeper functions.

## Performance in Practice

For typical reactive applications—user interfaces, data pipelines, live computations—the performance characteristics work out well. Most of the graph stays virtual, occupying only the memory of function closures. Materialization happens at branch points where multiple consumers share computed values. Updates propagate through minimal sets of tracked nodes.

The approach is particularly effective for read-heavy workloads. User interfaces spend most of their time reading values to render: each frame might read hundreds of observables but only change a handful. The virtual nodes give instant access without allocation, while materialized nodes provide cached results where sharing occurs.

Write-heavy scenarios or highly dynamic graphs might benefit from simpler lazy evaluation. If your application constantly rebuilds the dependency graph or updates most values on every change, the tracking overhead might dominate. But these cases are relatively rare in reactive applications where stability is the norm.

## Implementation Notes

FynX is built on a delta-based key-value store that maintains the dependency graph explicitly. The graph structure maintains:

- $\mathcal{G}$: forward edges (key → dependents)
- $\mathcal{G}^{-1}$: reverse edges (key → dependencies)
- $\text{indeg}$: indegree counts for topological sorting

When you create a computed observable, it registers its dependencies with the store. When a value changes, the store traverses $\mathcal{G}$ to find affected nodes and applies Kahn's algorithm for topological sorting.

Kahn's algorithm maintains a queue of nodes with zero in-degree within the affected subgraph:

$\text{Queue} \leftarrow \{n \in \text{Affected} : \text{indeg}_{\text{subgraph}}(n) = 0\}$

It processes nodes from the queue, reducing the in-degree of dependents and adding them when they reach zero. This guarantees that dependencies are computed before dependents, running in $O(|\text{Affected}| + |\text{edges}|)$ time—linear in the affected subgraph size.

Object pooling reduces allocation overhead for delta objects that represent changes. A simple pool maintains a list of reusable deltas, avoiding repeated allocation/deallocation during update propagation. Thread-local storage tracks computation context, particularly for cycle detection in the dependency graph.

Cycle detection happens through a simple visited set during computation. If a computed value tries to access itself (directly or transitively), the system detects the cycle and returns a cached value rather than infinitely recursing. This handles accidental cycles gracefully without sophisticated graph analysis.

## Design Philosophy

The core principle is to use structural information to guide caching decisions without manual annotation. Developers naturally express computation graphs through transforms and combinators. The patterns that emerge—linear chains, branch points, subscriptions—contain enough information to make good caching decisions automatically.

This differs from explicit hot/cold annotations where you mark certain observables as "signal" (always cache) or "stream" (never cache). Those systems push complexity to the developer: you need to understand the execution model well enough to annotate correctly. FynX moves that complexity into the runtime, using simple rules about fan-out and access patterns.

The approach has limits. It won't detect that a value deep in a virtual chain is accessed frequently by multiple unrelated computations. It won't evict unused materialized values to reclaim memory. These are opportunities for future enhancement, but they're not common enough to warrant the complexity for most applications.

## What You Get

The result is a reactive system where you build computation graphs naturally through transforms and combinators, and the system handles caching automatically. Linear chains stay lightweight. Branch points materialize shared values. Subscriptions work through explicit tracking. Changes propagate efficiently through minimal sets of affected nodes.

You can build complex reactive applications without thinking about hot versus cold observables, or which computations to cache. The structural patterns in your code contain enough information for the system to make good decisions. When performance matters, you still have control—forcing materialization or tracking is possible when needed.

Most importantly, the system scales gracefully. Adding more transforms doesn't linearly increase memory or computation overhead. Fusion keeps chains collapsed, and materialization happens only where sharing occurs. The result is reactive programming that feels natural to write and performs well without manual optimization.

## Further Directions

Several extensions could enhance the system while preserving its adaptive character. Runtime cost measurements could adjust the fan-out threshold dynamically—if a particular computation is expensive enough, caching after a single dependent might pay off. Cache eviction could reclaim memory from materialized nodes that haven't been accessed recently.

Incremental computation for collection operations would extend the approach to arrays and maps. When a collection changes, recompute only affected elements rather than the entire collection. This requires more sophisticated tracking but follows the same principle: use structural information to minimize work.

The fusion optimization could extend to more complex patterns beyond linear chains. Detecting common subexpressions across multiple computations could enable sharing at a finer granularity. But these enhancements should respect the core principle: automatic decisions based on structural patterns, not manual annotation.

## Conclusion

Adaptive materialization works because reactive computation graphs have regular structure. Linear chains dominate, and materialization at branch points captures the essential sharing. By watching how developers build their graphs and using simple rules about fan-out, the system achieves good performance without manual tuning.

The approach fits reactive programming naturally. You compose computations through transforms and combinators, expressing what you want rather than how to cache it. The system observes these patterns and materializes exactly where sharing occurs. The result is reactive programming that feels lightweight to write and performs well in practice.
