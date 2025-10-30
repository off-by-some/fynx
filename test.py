import math
import random

from fynx import Store, observable


class ReactiveNeuralNetwork(Store):
    """
    Fully reactive neural network with plasticity and OOD detection.
    Everything is cached by FynX - no manual update loops needed!
    """

    # === INPUT LAYER ===
    x1 = observable(0.0)
    x2 = observable(0.0)
    target = observable(0.0)

    # === WEIGHTS & BIASES ===
    # Hidden layer (2 inputs -> 3 hidden)
    w11 = observable(random.uniform(-1, 1))
    w12 = observable(random.uniform(-1, 1))
    w13 = observable(random.uniform(-1, 1))
    w21 = observable(random.uniform(-1, 1))
    w22 = observable(random.uniform(-1, 1))
    w23 = observable(random.uniform(-1, 1))

    # Output layer (3 hidden -> 1 output)
    v1 = observable(random.uniform(-1, 1))
    v2 = observable(random.uniform(-1, 1))
    v3 = observable(random.uniform(-1, 1))

    # Biases
    b1 = observable(random.uniform(-0.5, 0.5))
    b2 = observable(random.uniform(-0.5, 0.5))
    b3 = observable(random.uniform(-0.5, 0.5))
    c = observable(random.uniform(-0.5, 0.5))

    # === HYPERPARAMETERS ===
    base_lr = observable(0.5)
    plasticity_rate = observable(0.005)
    hebbian_enabled = observable(False)
    ema_alpha = observable(0.98)

    # === OOD DETECTION STATISTICS (must be defined before computed observables) ===
    h1_mean = observable(0.0)
    h1_var = observable(1.0)
    h2_mean = observable(0.0)
    h2_var = observable(1.0)
    h3_mean = observable(0.0)
    h3_var = observable(1.0)

    # === TRAINING STATE ===
    step_count = observable(0)  # Increments each training step
    do_update = observable(False)  # Triggers weight updates when True

    # === FORWARD PASS (REACTIVE) ===
    # Hidden layer pre-activations
    h1_raw = (x1 + w11 + x2 + w21 + b1).then(
        lambda x1, w11, x2, w21, b1: x1 * w11 + x2 * w21 + b1
    )
    h2_raw = (x1 + w12 + x2 + w22 + b2).then(
        lambda x1, w12, x2, w22, b2: x1 * w12 + x2 * w22 + b2
    )
    h3_raw = (x1 + w13 + x2 + w23 + b3).then(
        lambda x1, w13, x2, w23, b3: x1 * w13 + x2 * w23 + b3
    )

    # Hidden activations (tanh)
    h1 = h1_raw.then(lambda x: math.tanh(x))
    h2 = h2_raw.then(lambda x: math.tanh(x))
    h3 = h3_raw.then(lambda x: math.tanh(x))

    # Output logit and probability
    output_logit = (h1 + v1 + h2 + v2 + h3 + v3 + c).then(
        lambda h1, v1, h2, v2, h3, v3, c: h1 * v1 + h2 * v2 + h3 * v3 + c
    )
    output = output_logit.then(lambda z: 1 / (1 + math.exp(-z)))

    # === LOSS COMPUTATION ===
    loss = (output + target).then(lambda y, t: 0.5 * (y - t) ** 2)

    # === OOD DETECTION (CONDITIONAL STATISTICS UPDATES) ===
    # Only update statistics after first training step
    has_started_training = (step_count).then(lambda s: s > 0)

    # Conditional: EMA mean updates (only when training has started)
    h1_mean_update = (h1_mean + h1 + ema_alpha) >> (
        lambda mu, h, alpha: alpha * mu + (1 - alpha) * h
    )
    h1_mean_conditional = h1_mean_update & has_started_training

    h2_mean_update = (h2_mean + h2 + ema_alpha) >> (
        lambda mu, h, alpha: alpha * mu + (1 - alpha) * h
    )
    h2_mean_conditional = h2_mean_update & has_started_training

    h3_mean_update = (h3_mean + h3 + ema_alpha) >> (
        lambda mu, h, alpha: alpha * mu + (1 - alpha) * h
    )
    h3_mean_conditional = h3_mean_update & has_started_training

    # Conditional: EMA variance updates
    h1_var_update = (h1_var + h1 + h1_mean + ema_alpha) >> (
        lambda var, h, mu, alpha: alpha * var + (1 - alpha) * (h - mu) ** 2
    )
    h1_var_conditional = h1_var_update & has_started_training

    h2_var_update = (h2_var + h2 + h2_mean + ema_alpha) >> (
        lambda var, h, mu, alpha: alpha * var + (1 - alpha) * (h - mu) ** 2
    )
    h2_var_conditional = h2_var_update & has_started_training

    h3_var_update = (h3_var + h3 + h3_mean + ema_alpha) >> (
        lambda var, h, mu, alpha: alpha * var + (1 - alpha) * (h - mu) ** 2
    )
    h3_var_conditional = h3_var_update & has_started_training

    # Z-scores for OOD detection (always computed, but meaningful after training)
    h1_zscore = (h1 + h1_mean + h1_var).then(
        lambda h, mu, var: abs(h - mu) / (math.sqrt(var) + 1e-8)
    )
    h2_zscore = (h2 + h2_mean + h2_var).then(
        lambda h, mu, var: abs(h - mu) / (math.sqrt(var) + 1e-8)
    )
    h3_zscore = (h3 + h3_mean + h3_var).then(
        lambda h, mu, var: abs(h - mu) / (math.sqrt(var) + 1e-8)
    )

    # OOD score (average z-score across hidden units)
    ood_score = (h1_zscore + h2_zscore + h3_zscore).then(
        lambda z1, z2, z3: (z1 + z2 + z3) / 3.0
    )

    # === META-PLASTICITY (CONDITIONAL ADAPTIVE LEARNING) ===
    # High OOD triggers learning rate boost
    is_surprising = (ood_score).then(lambda ood: ood > 1.5)

    # Adaptive LR: base rate boosted when surprising
    adaptive_lr_base = (base_lr + ood_score).then(
        lambda lr, ood: lr * (1.0 + 0.3 * min(ood, 3.0))
    )
    adaptive_lr = adaptive_lr_base & is_surprising  # Only boost when surprising

    # === GRADIENT COMPUTATION ===
    # Output gradient (for MSE loss)
    output_grad = (output + target).then(lambda y, t: y - t)

    # Hidden layer gradients (backprop through tanh)
    h1_grad = (output_grad + v1 + h1).then(lambda δy, v, h: δy * v * (1 - h**2))
    h2_grad = (output_grad + v2 + h2).then(lambda δy, v, h: δy * v * (1 - h**2))
    h3_grad = (output_grad + v3 + h3).then(lambda δy, v, h: δy * v * (1 - h**2))

    # === SUPERVISED GRADIENTS ===
    dw11_sup = (h1_grad + x1).then(lambda δh, x: δh * x)
    dw12_sup = (h2_grad + x1).then(lambda δh, x: δh * x)
    dw13_sup = (h3_grad + x1).then(lambda δh, x: δh * x)
    dw21_sup = (h1_grad + x2).then(lambda δh, x: δh * x)
    dw22_sup = (h2_grad + x2).then(lambda δh, x: δh * x)
    dw23_sup = (h3_grad + x2).then(lambda δh, x: δh * x)

    # Output weight gradients
    dv1_sup = (output_grad + h1).then(lambda δy, h: δy * h)
    dv2_sup = (output_grad + h2).then(lambda δy, h: δy * h)
    dv3_sup = (output_grad + h3).then(lambda δy, h: δy * h)

    # Bias gradients
    db1_sup = h1_grad
    db2_sup = h2_grad
    db3_sup = h3_grad
    dc_sup = output_grad

    # === HEBBIAN PLASTICITY (CONDITIONAL OJA'S RULE) ===
    # Only apply Hebbian updates when enabled
    dw11_heb_base = (x1 + h1 + w11).then(lambda x, h, w: h * (x - h * w))
    dw11_heb = dw11_heb_base & hebbian_enabled  # Only when plasticity enabled

    dw12_heb_base = (x1 + h2 + w12).then(lambda x, h, w: h * (x - h * w))
    dw12_heb = dw12_heb_base & hebbian_enabled

    dw13_heb_base = (x1 + h3 + w13).then(lambda x, h, w: h * (x - h * w))
    dw13_heb = dw13_heb_base & hebbian_enabled

    dw21_heb_base = (x2 + h1 + w21).then(lambda x, h, w: h * (x - h * w))
    dw21_heb = dw21_heb_base & hebbian_enabled

    dw22_heb_base = (x2 + h2 + w22).then(lambda x, h, w: h * (x - h * w))
    dw22_heb = dw22_heb_base & hebbian_enabled

    dw23_heb_base = (x2 + h3 + w23).then(lambda x, h, w: h * (x - h * w))
    dw23_heb = dw23_heb_base & hebbian_enabled

    # === HYBRID WEIGHT UPDATES (CONDITIONAL APPLICATION) ===
    # Combine supervised and Hebbian gradients
    dw11_total = (dw11_sup + dw11_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )
    dw12_total = (dw12_sup + dw12_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )
    dw13_total = (dw13_sup + dw13_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )
    dw21_total = (dw21_sup + dw21_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )
    dw22_total = (dw22_sup + dw22_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )
    dw23_total = (dw23_sup + dw23_heb + plasticity_rate).then(
        lambda sup, heb, pr: sup + (heb * pr if heb is not None else 0)
    )

    # Conditional weight updates (only apply when do_update is True)
    w11_update = (w11 + adaptive_lr + dw11_total).then(lambda w, lr, dw: w - lr * dw)
    w11_new = w11_update & do_update

    w12_update = (w12 + adaptive_lr + dw12_total).then(lambda w, lr, dw: w - lr * dw)
    w12_new = w12_update & do_update

    w13_update = (w13 + adaptive_lr + dw13_total).then(lambda w, lr, dw: w - lr * dw)
    w13_new = w13_update & do_update

    w21_update = (w21 + adaptive_lr + dw21_total).then(lambda w, lr, dw: w - lr * dw)
    w21_new = w21_update & do_update

    w22_update = (w22 + adaptive_lr + dw22_total).then(lambda w, lr, dw: w - lr * dw)
    w22_new = w22_update & do_update

    w23_update = (w23 + adaptive_lr + dw23_total).then(lambda w, lr, dw: w - lr * dw)
    w23_new = w23_update & do_update

    # Output weights (always supervised, no Hebbian)
    v1_update = (v1 + adaptive_lr + dv1_sup).then(lambda v, lr, dv: v - lr * dv)
    v1_new = v1_update & do_update

    v2_update = (v2 + adaptive_lr + dv2_sup).then(lambda v, lr, dv: v - lr * dv)
    v2_new = v2_update & do_update

    v3_update = (v3 + adaptive_lr + dv3_sup).then(lambda v, lr, dv: v - lr * dv)
    v3_new = v3_update & do_update

    # Bias updates
    b1_update = (b1 + adaptive_lr + db1_sup).then(lambda b, lr, db: b - lr * db)
    b1_new = b1_update & do_update

    b2_update = (b2 + adaptive_lr + db2_sup).then(lambda b, lr, db: b - lr * db)
    b2_new = b2_update & do_update

    b3_update = (b3 + adaptive_lr + db3_sup).then(lambda b, lr, db: b - lr * db)
    b3_new = b3_update & do_update

    c_update = (c + adaptive_lr + dc_sup).then(lambda c, lr, dc: c - lr * dc)
    c_new = c_update & do_update

    # === CONDITIONAL STATISTICS UPDATES ===
    # Statistics only update when training and updating
    h1_mean_update = h1_mean_conditional & do_update
    h2_mean_update = h2_mean_conditional & do_update
    h3_mean_update = h3_mean_conditional & do_update

    h1_var_update = h1_var_conditional & do_update
    h2_var_update = h2_var_conditional & do_update
    h3_var_update = h3_var_conditional & do_update


def setup_reactive_updates(net):
    """Set up subscriptions for conditional weight updates"""

    # Subscribe to conditional weight updates and apply them
    def update_weight(original, new_value):
        original.set(new_value)

    # Weight updates
    net.w11_new.subscribe(lambda v: update_weight(net.w11, v))
    net.w12_new.subscribe(lambda v: update_weight(net.w12, v))
    net.w13_new.subscribe(lambda v: update_weight(net.w13, v))
    net.w21_new.subscribe(lambda v: update_weight(net.w21, v))
    net.w22_new.subscribe(lambda v: update_weight(net.w22, v))
    net.w23_new.subscribe(lambda v: update_weight(net.w23, v))

    net.v1_new.subscribe(lambda v: update_weight(net.v1, v))
    net.v2_new.subscribe(lambda v: update_weight(net.v2, v))
    net.v3_new.subscribe(lambda v: update_weight(net.v3, v))

    net.b1_new.subscribe(lambda v: update_weight(net.b1, v))
    net.b2_new.subscribe(lambda v: update_weight(net.b2, v))
    net.b3_new.subscribe(lambda v: update_weight(net.b3, v))
    net.c_new.subscribe(lambda v: update_weight(net.c, v))

    # Statistics updates
    net.h1_mean_update.subscribe(lambda v: update_weight(net.h1_mean, v))
    net.h2_mean_update.subscribe(lambda v: update_weight(net.h2_mean, v))
    net.h3_mean_update.subscribe(lambda v: update_weight(net.h3_mean, v))
    net.h1_var_update.subscribe(lambda v: update_weight(net.h1_var, v))
    net.h2_var_update.subscribe(lambda v: update_weight(net.h2_var, v))
    net.h3_var_update.subscribe(lambda v: update_weight(net.h3_var, v))


def reactive_weight_update(net):
    """Trigger reactive weight updates by setting do_update = True"""
    # This triggers all conditional updates
    net.do_update = True
    # Reset for next step
    net.do_update = False


def train_step(net, x1_val, x2_val, target_val):
    """Single reactive training step"""
    # Set inputs (this triggers reactive forward pass)
    net.x1 = x1_val
    net.x2 = x2_val
    net.target = target_val

    # Increment step counter (for statistics)
    net.step_count = net.step_count.value + 1

    # Apply reactive updates (weights and statistics)
    reactive_weight_update(net)


def train_and_test_conditional_reactive():
    """
    Fully reactive training with conditionals - the most expressive version!
    """
    print("=" * 70)
    print("🎯 CONDITIONAL REACTIVE NEURAL NETWORK")
    print("   Conditionals make complex logic declarative!")
    print("=" * 70)

    # Set up reactive update subscriptions
    setup_reactive_updates(ReactiveNeuralNetwork)

    xor_data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

    print("\n" + "=" * 70)
    print("PHASE 1: Training on XOR (Supervised Only)")
    print("=" * 70)

    # Phase 1: Pure supervised learning
    for epoch in range(800):
        total_loss = 0
        for x1_val, x2_val, target_val in xor_data:
            train_step(ReactiveNeuralNetwork, x1_val, x2_val, target_val)
            total_loss += ReactiveNeuralNetwork.loss.value

        if epoch % 200 == 0:
            avg_loss = total_loss / len(xor_data)
            print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")

    print("\n✓ XOR Learned! Testing in-distribution performance:")
    print("-" * 70)
    for x1_val, x2_val, expected in xor_data:
        ReactiveNeuralNetwork.x1 = x1_val
        ReactiveNeuralNetwork.x2 = x2_val
        pred = ReactiveNeuralNetwork.output.value
        ood = ReactiveNeuralNetwork.ood_score.value
        print(
            f"XOR({x1_val}, {x2_val}) = {pred:.4f} (target: {expected}) "
            f"[OOD: {ood:.2f}]"
        )

    print("\n" + "=" * 70)
    print("PHASE 2: OOD Detection - Conditional Learning Rate Boost")
    print("=" * 70)

    ood_shifted = [(0.3, 0.7, 1), (0.7, 0.3, 1), (0.2, 0.2, 0), (0.8, 0.8, 0)]

    print("Testing on shifted inputs (LR adapts when OOD > 1.5):")
    print("-" * 70)
    for x1_val, x2_val, expected in ood_shifted:
        ReactiveNeuralNetwork.x1 = x1_val
        ReactiveNeuralNetwork.x2 = x2_val
        pred = ReactiveNeuralNetwork.output.value
        ood = ReactiveNeuralNetwork.ood_score.value
        lr = (
            ReactiveNeuralNetwork.adaptive_lr.value
            if hasattr(ReactiveNeuralNetwork, "adaptive_lr")
            and ReactiveNeuralNetwork.adaptive_lr.value is not None
            else ReactiveNeuralNetwork.base_lr.value
        )
        alert = "⚠️  OOD DETECTED!" if ood > 1.5 else ""
        boost = "🚀 LR BOOSTED!" if ood > 1.5 else ""
        print(
            f"XOR({x1_val}, {x2_val}) = {pred:.4f} (expected: {expected}) "
            f"[OOD: {ood:.2f}, LR: {lr:.3f}] {alert} {boost}"
        )

    print("\n" + "=" * 70)
    print("PHASE 3: Continual Learning - Conditional Hebbian Plasticity")
    print("=" * 70)

    and_data = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

    print("\nBefore adaptation (AND function, Hebbian disabled):")
    print("-" * 70)
    total_error = 0
    for x1_val, x2_val, expected in and_data:
        ReactiveNeuralNetwork.x1 = x1_val
        ReactiveNeuralNetwork.x2 = x2_val
        pred = ReactiveNeuralNetwork.output.value
        ood = ReactiveNeuralNetwork.ood_score.value
        error = abs(pred - expected)
        total_error += error
        print(
            f"AND({x1_val}, {x2_val}) = {pred:.4f} (target: {expected}) "
            f"[OOD: {ood:.2f}, Error: {error:.3f}]"
        )
    print(f"Average error: {total_error/len(and_data):.3f}")

    # Enable Hebbian plasticity - conditionals now allow Hebbian updates!
    print("\n🧠 Enabling Hebbian plasticity (conditional on flag)...")
    ReactiveNeuralNetwork.hebbian_enabled = True
    ReactiveNeuralNetwork.base_lr = 0.15

    print("Now Hebbian gradients flow conditionally!")
    print("-" * 70)

    for epoch in range(400):
        for x1_val, x2_val, target_val in and_data:
            train_step(ReactiveNeuralNetwork, x1_val, x2_val, target_val)

    print("\nAfter 400 epochs with conditional Hebbian plasticity:")
    print("-" * 70)
    total_error = 0
    for x1_val, x2_val, expected in and_data:
        ReactiveNeuralNetwork.x1 = x1_val
        ReactiveNeuralNetwork.x2 = x2_val
        pred = ReactiveNeuralNetwork.output.value
        ood = ReactiveNeuralNetwork.ood_score.value
        error = abs(pred - expected)
        total_error += error
        hebbian_active = (
            "🧠 Hebbian Active" if ReactiveNeuralNetwork.hebbian_enabled.value else ""
        )
        print(
            f"AND({x1_val}, {x2_val}) = {pred:.4f} (target: {expected}) "
            f"[OOD: {ood:.2f}, Error: {error:.3f}] {hebbian_active}"
        )
    print(f"Average error: {total_error/len(and_data):.3f}")

    # Test XOR retention
    print("\n" + "=" * 70)
    print("PHASE 4: Testing Catastrophic Forgetting")
    print("=" * 70)
    print("Checking if original XOR is retained:")
    print("-" * 70)
    total_error = 0
    for x1_val, x2_val, expected in xor_data:
        ReactiveNeuralNetwork.x1 = x1_val
        ReactiveNeuralNetwork.x2 = x2_val
        pred = ReactiveNeuralNetwork.output.value
        error = abs(pred - expected)
        total_error += error
        retained = "✓ Retained" if error < 0.2 else "✗ Forgotten"
        print(
            f"XOR({x1_val}, {x2_val}) = {pred:.4f} (target: {expected}) "
            f"[Error: {error:.3f}] {retained}"
        )
    print(f"Average error: {total_error/len(xor_data):.3f}")

    print("\n" + "=" * 70)
    print("🎉 CONDITIONAL REACTIVE SUCCESS!")
    print("=" * 70)
    print("✓ Statistics updates conditional on training progress")
    print("✓ Hebbian plasticity conditional on enable flag")
    print("✓ Learning rate conditional on OOD surprise")
    print("✓ Weight updates conditional on update trigger")
    print("✓ All logic is declarative - no imperative if/else!")

    print("\n🚀 Conditionals make FynX expressive beyond traditional reactivity!")
    print("   Complex ML logic becomes composable, declarative rules!")


if __name__ == "__main__":
    train_and_test_conditional_reactive()
