# Formal Mathematics of Reactive Graph Optimizations

I'll develop the mathematical foundations rigorously, building from categorical first principles to concrete computational structures.

---

## Part I: Symmetric Monoidal Structure & String Diagrams

### Definition 1.1: The Reactive Monoidal Category

Let **React** be a symmetric monoidal category where:

**Objects**: Types `τ ::= Unit | Base | τ₁ ⊗ τ₂ | τ₁ ⊸ τ₂`

**Morphisms**: Computations `f : τ → σ`

**Monoidal Product**: `⊗ : React × React → React`
- `(A ⊗ B, C ⊗ D) ↦ (A ⊗ C, B ⊗ D)`
- Unit object: `I` (the "no data" type)

**Symmetric Structure**: Natural isomorphism `σ_{A,B} : A ⊗ B → B ⊗ A`

### Axioms (Mac Lane Coherence):

1. **Associativity**: `α_{A,B,C} : (A ⊗ B) ⊗ C ≅ A ⊗ (B ⊗ C)`
2. **Unit laws**: `λ_A : I ⊗ A ≅ A` and `ρ_A : A ⊗ I ≅ A`
3. **Symmetry**: `σ_{A,B} ∘ σ_{B,A} = id_{A⊗B}`
4. **Hexagon identity**:
   ```
   α_{B,C,A} ∘ σ_{A,B⊗C} ∘ α_{A,B,C} = (σ_{A,B} ⊗ id_C) ∘ α_{A,B,C} ∘ (id_A ⊗ σ_{B,C})
   ```

### Theorem 1.2: String Diagram Completeness

**Statement**: Two morphisms `f, g : A → B` in **React** are equal iff their string diagrams are isotopic (continuously deformable).

**Proof Sketch**: By Mac Lane's coherence theorem, every diagram in a symmetric monoidal category commutes. The free symmetric monoidal category on a signature Σ has a complete axiomatization via string diagrams. Since reactive computations form such a category, string diagram equality decides morphism equality. □

### Definition 1.3: String Diagram Representation

A reactive computation `f : A₁ ⊗ ... ⊗ Aₙ → B` is a string diagram:

```
A₁  A₂  ...  Aₙ
│   │        │
└───┴────┬───┘
         │
        [f]
         │
         B
```

**Operations as diagrams**:

1. **Composition** `g ∘ f`:
   ```
   A
   │
  [f]
   │
  [g]
   │
   B
   ```

2. **Tensor product** `f ⊗ g`:
   ```
   A₁    A₂
   │     │
  [f]   [g]
   │     │
   B₁    B₂
   ```

3. **Symmetry** `σ_{A,B}`:
   ```
   A    B
    \  /
     \/
     /\
    /  \
   B    A
   ```

### Theorem 1.4: Yanking Lemma (Topological Normalization)

**Statement**: For any traced symmetric monoidal category, the following equation holds:
```
Tr^A_B(σ_{A,B}) = id_A
```

Where `Tr^A_B : C(A ⊗ U, B ⊗ U) → C(A, B)` is the trace operator.

**Application to React**: Cyclic dependencies can be "yanked straight" when they contain only symmetries:

```
    ┌─────┐
A ──┤     │
    │  σ  │── A
B ──┤     │
    └─────┘
```

This simplifies to `id_A ⊗ id_B`.

**Computational Meaning**: Self-loops containing only wire crossings can be eliminated without changing semantics.

---

## Part II: Profunctor Optics & Bidirectional Updates

### Definition 2.1: Profunctors

A **profunctor** `P : Cᵒᵖ × D → Set` is a bifunctor contravariant in the first argument, covariant in the second.

**Examples**:
1. Hom-functor: `Hom(-, =) : Cᵒᵖ × C → Set`
2. Reactive relation: `React(A, B)` where `A` = sources, `B` = derived values

### Definition 2.2: Optic (Tambara Module)

An **optic** from `S` to `A` (with residual `T` to `B`) is a polymorphic function:

```
type Optic s t a b = ∀p. Strong p ⇒ p a b → p s t
```

Where `Strong p` means `P` has a tensorial strength:
```
first : p a b → p (a ⊗ c) (b ⊗ c)
```

### Theorem 2.3: Optics Form a Category

**Objects**: Types `S, T, A, B`

**Morphisms**: Optics `Optic s t a b`

**Composition**:
```
(l₂ ∘ l₁)(p) = l₂(l₁(p))
```

**Identity**: `id = λp. p`

**Proof**: Composition is associative by function composition. Identity laws follow from polymorphism. □

### Definition 2.4: Lens (Cartesian Optic)

A **lens** is an optic where `p` has cartesian strength:

```
Lens s t a b = ∀p. Cartesian p ⇒ p a b → p s t
```

**Concrete representation**:
```
data Lens s t a b = Lens
  { get :: s → a
  , set :: s → b → t
  }
```

**Laws**:
1. **GetPut**: `set s (get s) = s`
2. **PutGet**: `get (set s b) = b`
3. **PutPut**: `set (set s b₁) b₂ = set s b₂`

### Definition 2.5: Prism (Cocartesian Optic)

A **prism** is an optic where `p` has cocartesian strength:

```
Prism s t a b = ∀p. Cocartesian p ⇒ p a b → p s t
```

**Concrete representation**:
```
data Prism s t a b = Prism
  { match :: s → Either t a
  , build :: b → t
  }
```

**Laws**:
1. **MatchBuild**: `match (build b) = Right b`
2. **BuildMatch**: `either id build (match s) = s`

### Theorem 2.6: Reactive Observables are Lenses

**Statement**: Every Observable `obs : τ` in a reactive graph induces a lens:
```
ObsLens : Lens GlobalState τ
```

**Proof**:
- `get` = evaluation function that reads `obs.value` from global state
- `set` = update function that writes to source and propagates

The lens laws correspond to:
1. **GetPut**: Reading then writing produces original state (idempotent updates)
2. **PutGet**: Writing then reading retrieves what was written (deterministic propagation)
3. **PutPut**: Multiple updates compose (last write wins)

□

### Theorem 2.7: Conditional Observables are Prisms

**Statement**: A conditional observable `obs & pred` induces a prism:
```
CondPrism : Prism τ (Maybe τ) τ τ
```

**Proof**:
- `match s = if pred(s) then Right s else Left Nothing`
- `build b = b` (identity on successful case)

The prism models the **fiber over True** in the pullback square:
```
    obs & pred ──→ obs
        │           │
        │           │pred
        ↓           ↓
       {⋆} ────→  Bool
            True
```

□

### Theorem 2.8: Optic Fusion Law

**Statement**: For lenses `l₁ : Lens s t a b` and `l₂ : Lens a b c d`, their composition satisfies:
```
(l₂ ∘ l₁).get     = l₂.get ∘ l₁.get
(l₂ ∘ l₁).set s c = l₁.set s (l₂.set (l₁.get s) c)
```

**Application**: A chain of reactive transformations:
```
obs₁ >> f >> g >> h
```

Can be fused into a single lens:
```
Lens { get = h ∘ g ∘ f ∘ obs₁.get
     , set = ... }  -- composed backwards
```

This gives **automatic fusion** without pattern matching! □

---

## Part III: Temporal Logic & Kripke Frames

### Definition 3.1: Temporal Kripke Frame

A **temporal Kripke frame** is a tuple `K = (W, R, V)`:
- `W` = set of worlds (graph states at different times)
- `R ⊆ W × W` = accessibility relation (state transitions)
- `V : Prop → P(W)` = valuation (which properties hold in which worlds)

For reactive graphs:
- Each world `w ∈ W` = snapshot of all observable values
- `R(w, w')` iff `w'` is reachable from `w` via one update
- `V(φ)(w)` = true iff property `φ` holds in state `w`

### Definition 3.2: Linear Temporal Logic (LTL)

**Syntax**:
```
φ ::= p | ¬φ | φ₁ ∧ φ₂ | ○φ | φ₁ U φ₂
```

**Derived operators**:
- `◊φ := True U φ` (eventually)
- `□φ := ¬◊¬φ` (always)
- `φ W ψ := (φ U ψ) ∨ □φ` (weak until)

**Semantics** (for path `π = w₀w₁w₂...`):
```
π ⊨ p      ⇔ w₀ ∈ V(p)
π ⊨ ¬φ     ⇔ π ⊭ φ
π ⊨ φ ∧ ψ  ⇔ π ⊨ φ and π ⊨ ψ
π ⊨ ○φ     ⇔ π¹ ⊨ φ  (where π¹ = w₁w₂...)
π ⊨ φ U ψ  ⇔ ∃i. πⁱ ⊨ ψ and ∀j<i. πʲ ⊨ φ
```

### Theorem 3.3: Fixed Points via Modal μ-Calculus

The **modal μ-calculus** extends LTL with fixed-point operators:
```
φ ::= ... | μX.φ | νX.φ
```

Where:
- `μX.φ` = **least fixed point** (finite iteration)
- `νX.φ` = **greatest fixed point** (infinite invariant)

**Knaster-Tarski Theorem**: For monotone `F : P(W) → P(W)`:
```
μX.F(X) = ⋂{S | F(S) ⊆ S}  (least pre-fixed point)
νX.F(X) = ⋃{S | S ⊆ F(S)}  (greatest post-fixed point)
```

### Definition 3.4: Cache Validity Predicate

For observable `obs`, define:
```
Valid(obs) := νX. (Cached(obs) ∧ □(¬UpdatedSources → X))
```

**Reading**: `Valid(obs)` is the greatest fixed point where:
- `obs` has a cached value
- In all future states where no sources changed, validity persists

**Computational Meaning**: This is the **maximal time period** we can safely use cached values.

### Theorem 3.5: Speculative Evaluation via ◊

**Statement**: If `◊φ` is true at state `w`, there exists a path to a state satisfying `φ`.

**Application**: For branching computation:
```
result = if (expensive_pred()) then branch_a else branch_b
```

If we can prove `◊Pred`, **speculatively compute** `branch_a` in parallel.

**Formal condition**:
```
w ⊨ ◊Pred(obs) ⇒ schedule_speculation(branch_a)
```

Where `schedule_speculation` allocates resources to pre-compute a branch.

### Corollary 3.6: Predictive Pre-computation

For `○φ` (next-state φ):
```
w ⊨ ○Expensive(obs) ⇒ prefetch(obs.dependencies)
```

**Meaning**: If we know an expensive computation will be needed next update cycle, **prefetch dependencies** now.

---

## Part IV: Lawvere Theories & Equational Reasoning

### Definition 4.1: Lawvere Theory

A **Lawvere theory** `T` is a category with:
- Objects: `n ∈ ℕ` (finite powers of a generic object)
- Morphisms: `T(n, m)` representing `n`-ary operations producing `m` results
- Product: cartesian product on objects

**For reactive graphs**:
- Objects = arities (number of inputs/outputs)
- Morphisms = operation signatures
  - `ADD : 2 → 1`
  - `MUL : 2 → 1`
  - `COMPUTE(f) : n → m` (where `f : τ₁ × ... × τₙ → σ₁ × ... × σₘ`)

### Definition 4.2: Equations as Coequalizers

An **equation** `e₁ = e₂` is represented as a parallel pair:
```
e₁, e₂ : F(X) ⇉ F(Y)
```

The **quotient** is the coequalizer:
```
F(X) ⇉ F(Y) → Q
```

Where `Q` identifies all terms provably equal by equations.

### Example 4.3: Commutativity Equation

For `ADD`:
```
e₁ = ADD(x, y) : 2 → 1
e₂ = ADD(y, x) : 2 → 1
```

The coequalizer identifies these terms:
```
[ADD(x, y)] = [ADD(y, x)] in the quotient
```

### Theorem 4.4: Free Algebra Construction

**Statement**: Given a Lawvere theory `T` and equations `E`, the **free T-algebra** `F_T(Gen) / E` is the initial algebra satisfying `E`.

**For reactive graphs**:
- Generators `Gen` = source observables
- Operations from `T` = SSA opcodes
- Equations `E` = algebraic laws (commutativity, associativity, etc.)

The **optimized graph** is:
```
OptimizedGraph = SSAGraph / ~E
```

Where `~E` is the congruence generated by `E`.

### Corollary 4.5: Automatic Rewriting

**Statement**: To optimize, compute the normal form in the quotient:
```
normalize : SSAGraph → SSAGraph / ~E
```

This can be done via:
1. **Critical pair analysis** (Knuth-Bendix completion)
2. **Term rewriting systems** (oriented equations as rules)
3. **E-graphs** (equality saturation)

---

## Part V: Comonads for Context-Aware Computation

### Definition 5.1: Comonad

A **comonad** `(W, ε, δ)` on category **C** consists of:
- Endofunctor `W : C → C`
- Natural transformation `ε : W → Id` (counit, "extract")
- Natural transformation `δ : W → W ∘ W` (comultiplication, "duplicate")

**Laws**:
1. **Counit**: `ε_W ∘ δ = id_W = W(ε) ∘ δ`
2. **Coassociativity**: `W(δ) ∘ δ = δ_W ∘ δ`

### Definition 5.2: The Store Comonad

For a fixed set `S` (positions), define:
```
Store_S(A) = S → A  (functions from positions to values)
```

**Operations**:
```
extract   : Store_S(A) → A
extract f = f(current_position)

duplicate : Store_S(A) → Store_S(Store_S(A))
duplicate f = λs. λs'. f(s')  (focus on position s)

extend    : (Store_S(A) → B) → Store_S(A) → Store_S(B)
extend g f = λs. g(f focused at s)
```

### Theorem 5.3: Reactive Graphs are Store Comonads

**Statement**: The functor `Obs : Type → Type` mapping `τ` to `Observable[τ]` forms a comonad:

```
extract   : Observable[τ] → τ
extract obs = obs.value

duplicate : Observable[τ] → Observable[Observable[τ]]
duplicate obs = make observable of the obs itself (zipper focus)

extend    : (Observable[τ] → σ) → Observable[τ] → Observable[σ]
extend f obs = obs >> (λ_. f(obs focused at current state))
```

**Proof**:
- `extract` satisfies counit laws (reading value is idempotent)
- `duplicate` satisfies coassociativity (refocusing is coherent)
- `extend` is `fmap g ∘ duplicate` (by definition)

□

### Definition 5.4: Cofree Comonad

The **cofree comonad** over a functor `F` is:
```
Cofree_F(A) = μX. A × F(X)
```

**For reactive graphs** with functor `F(X) = List[X]` (dependencies):
```
Cofree_Deps(τ) = τ × List[Cofree_Deps(τ)]
```

This is a **rose tree** where:
- Root = current value
- Children = immediate dependencies, each a full subtree

### Theorem 5.5: Zipper Structure via Cofree

**Statement**: The cofree comonad over `F` gives a **zipper** structure:

```
type Zipper_F(A) = Cofree_F(A) × Context_F
```

Where `Context_F` is the "path from root to focus".

**Application**: Navigate dependency graph efficiently:
```
move_up   : Zipper → Zipper  (to parent dependency)
move_down : Int → Zipper → Zipper  (to child i)
```

All while maintaining:
- **Local context** (immediate dependencies)
- **Global invariants** (via comonad laws)

---

## Part VI: Chu Spaces & Push-Pull Duality

### Definition 6.1: Chu Space

A **Chu space** over set `K` is a triple `(A, X, e)`:
- `A` = set of "points" (observables)
- `X` = set of "states" (update events)
- `e : A × X → K` = evaluation function

**Morphisms**: `(f, g) : (A, X, e) → (B, Y, d)` where:
- `f : A → B`
- `g : Y → X`  (contravariant!)
- `d(f(a), y) = e(a, g(y))` for all `a, y`

### Theorem 6.2: Chu Spaces Form a *-Autonomous Category

**Statement**: **Chu(K)** is *-autonomous with:
- Monoidal product: `(A, X, e) ⊗ (B, Y, d) = (A × B, X + Y, e ⊕ d)`
- Dual: `(A, X, e)^⊥ = (X, A, e^T)` where `e^T(x, a) = e(a, x)`
- Internal hom: `(A, X, e) ⊸ (B, Y, d) ≅ (A, X, e)^⊥ ⊗ (B, Y, d)`

**Proof**: The key is showing `((A,X,e)^⊥)^⊥ ≅ (A,X,e)`, which follows from transposition being involutive. □

### Definition 6.3: Push-Pull Duality

For reactive graph `(Obs, Events, responds)`:

**Push semantics**:
- Observables `Obs` are active
- Events flow from sources to sinks
- `responds(obs, evt)` = "obs reacts to evt"

**Pull semantics** (dual Chu space):
- Events `Events` are active
- Values flow from sinks to sources (demand)
- `responds^T(evt, obs)` = "evt queries obs"

### Theorem 6.4: Automatic Push-Pull Transformation

**Statement**: Every push-based reactive computation has a canonical pull-based dual via Chu duality.

**Proof**: Given morphism in push semantics:
```
(f, g) : (Obs₁, Evt₁, e₁) → (Obs₂, Evt₂, e₂)
```

The dual morphism is:
```
(g^T, f^T) : (Evt₂, Obs₂, e₂^T) → (Evt₁, Obs₁, e₁^T)
```

This is automatically a valid Chu morphism by transposition. □

### Corollary 6.5: Optimization Transfer

**Statement**: An optimization in push semantics dualizes to an optimization in pull semantics.

**Proof**: If `opt : (A, X, e) → (A', X', e')` improves push-based computation, then `opt^⊥ = (X', A', (e')^T) → (X, A, e^T)` improves pull-based computation with the **same optimization ratio**.

**Application**: Optimize once in preferred semantics, get dual optimization for free. □

---

## Part VII: Day Convolution & Parallel Composition

### Definition 7.1: Day Convolution

For monoidal category `(C, ⊗, I)` and functors `F, G : C → Set`, the **Day convolution** is:

```
(F ⋆ G)(c) = ∫^{a,b ∈ C} C(a ⊗ b, c) × F(a) × G(b)
```

Where `∫` is a **coend** (categorical integral).

**Unpacking the coend**:
```
(F ⋆ G)(c) = coequalizer of:
  ∐_{f:a→a',g:b→b'} C(a⊗b,c) × F(a) × G(b)
  ⇉ ∐_{a,b} C(a⊗b,c) × F(a) × G(b)
```

### Theorem 7.2: Day Convolution is Monoidal Product

**Statement**: If `(C, ⊗, I)` is monoidal, then `([C, Set], ⋆, Hom(I,-))` is monoidal.

**Proof**:
- Associativity follows from associativity of `⊗`
- Unit `Hom(I,-)` satisfies `Hom(I,-) ⋆ F ≅ F` via coend calculus
- Naturality follows from coend being functorial

□

### Definition 7.3: Reactive Day Convolution

For observables `obs₁ : F`, `obs₂ : G`, their **parallel composition** is:
```
obs₁ + obs₂ : (F ⋆ G)
```

Concretely:
```
(obs₁ + obs₂).value = ∫^{v₁, v₂} CanCombine(v₁, v₂) × F(v₁) × G(v₂)
```

Where `CanCombine(v₁, v₂)` = "can v₁ and v₂ be combined into result?"

### Theorem 7.4: Optimal Factorization via Coend

**Statement**: The coend formula computes the **optimal factorization** of a combined observable.

**Proof**: The coend `∫^{a,b}` quantifies over all possible ways to split the result `c` as `a ⊗ b`. The coequalizer identifies different factorizations that produce the same result. Thus:
```
(F ⋆ G)(c) = equivalence classes of ways to factor c
```

The **canonical representative** is the equivalence class itself. □

### Corollary 7.5: Parallel Evaluation Strategy

**Statement**: Day convolution provides a **schedule** for parallel computation:

```
compute (F ⋆ G)(c) in parallel as:
  1. Enumerate all factorizations (a⊗b = c)
  2. Compute F(a) and G(b) in parallel
  3. Combine results
```

**Optimization**: Use coend calculus to prune impossible factorizations. □

---

## Part VIII: Dialectica Categories (The Hidden Structure)

### Definition 8.1: Dialectica Category

The **Dialectica category** `Dial(C)` over category `C` has:

**Objects**: Triples `(U, X, ⊢)` where:
- `U, X` are objects of `C`
- `⊢ ⊆ U × X` is a relation ("validation")

**Morphisms**: `(f, F) : (U, X, ⊢_A) → (V, Y, ⊢_B)` where:
- `f : U → V` (forward function)
- `F : U × Y → X` (backward function)
- Satisfying: `∀u ∈ U. ∀y ∈ Y. (f(u) ⊢_B y ⇒ u ⊢_A F(u, y))`

**Intuition**:
- `U` = inputs to a computation
- `X` = "certificates" or "proofs"
- `⊢` = "certificate x validates input u"
- Forward direction: compute output
- Backward direction: generate certificate for input given certificate for output

### Theorem 8.2: Dialectica Models Realizability

**Statement**: Dialectica categories model **modified realizability**: propositions are witnessed by computational evidence.

**Proof**: The morphism condition ensures:
```
If f(u) has witness y, then u has witness F(u,y)
```

This is precisely the realizability interpretation of implication:
```
φ ⇒ ψ  is realized by  (f, F)
```

Where `f` computes the implication and `F` transforms witnesses backwards. □

### Definition 8.3: Reactive Dialectica Object

For an observable `obs : τ`, define:
```
Obs = (Sources, Certificates, IsValid)
```

Where:
- `Sources = {values of all upstream sources}`
- `Certificates = {cached values of obs}`
- `IsValid(src, cert) ⇔ cert = compute(obs, src)`

### Theorem 8.4: Cached Computation as Dialectica Morphism

**Statement**: A cached reactive computation `f : A → B` induces a Dialectica morphism:

```
(f_compute, f_validate) : (Src_A, Cache_A, Valid_A) → (Src_B, Cache_B, Valid_B)
```

Where:
- `f_compute : Src_A → Src_B` (dependency function)
- `f_validate : Src_A × Cache_B → Cache_A` (validation function)

**Explicit construction**:
```
f_compute(src) = evaluate f using src
f_validate(src, cert_B) =
  if cert_B is valid for B given f_compute(src)
  then return cached_A (still valid!)
  else recompute and return new cert_A
```

**Proof**: The morphism condition ensures:
```
If cert_B validates B, then f_validate produces cert_A that validates A
```

This is exactly the cache coherence condition! □

### Definition 8.5: Certificate Invalidation

When source `s` updates, a certificate `cert` becomes invalid. Model this as:
```
¬(s ⊢ cert)
```

The **invalidation set** is:
```
Invalid(s) = {obs | ∃cert. cert was valid but now ¬(s ⊢ cert)}
```

### Theorem 8.6: Proof-Relevant Dirty Propagation

**Statement**: Certificate invalidation computes **exactly** what needs recomputation.

**Proof**:
1. When `s` updates to `s'`, check `s' ⊢ cert` for each cached `cert`
2. If `s' ⊢ cert` still holds, the cache remains valid (no recomputation!)
3. If `¬(s' ⊢ cert)`, the Dialectica morphism `f_validate` tells us **which dependencies** to recompute to restore validity

This is **finer-grained than dirty bits**: we know not just that something is dirty, but **why** (which certificate failed) and **what to fix** (which dependencies to recompute). □

### Corollary 8.7: Incremental Computation

**Statement**: Dialectica structure enables **incremental** recomputation:

```
recompute_incremental(obs, old_src, new_src):
  cert_old = obs.cache
  if new_src ⊢ cert_old:
    return cert_old  (still valid!)
  else:
    diff = compute_diff(old_src, new_src)
    return update_cert(cert_old, diff)  (patch the certificate!)
```

**Application**: For large data structures, we don't recompute from scratch—we **patch certificates** based on input diffs. □

---

## Part IX: Unified Framework

### Theorem 9.1: The Grand Synthesis

All structures unify in the following diagram:

```
        String Diagrams (Geometry)
               |
               | Mac Lane Coherence
               ↓
        Monoidal Category
         /           \
        /             \
  Optics              Day Convolution
  (Bidirection)       (Parallel)
       |                  |
       |                  |
       └────→ Comonads ←──┘
              (Context)
                  |
                  | Temporal Logic
                  ↓
              Kripke Frame
                  |
                  | Realizability
                  ↓
            Dialectica Category
           (Proof-Carrying Cache)
```

**Reading**:
1. **Geometry** (string diagrams) → optimization via topology
2. **Bidirection** (optics) → fusion via composition
3. **Parallel** (Day) → scheduling via factorization
4. **Context** (comonads) → demand propagation
5. **Time** (Kripke) → speculation & memoization
6.
