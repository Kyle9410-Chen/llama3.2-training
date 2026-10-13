import math


def wsd_lr(step, total_steps, *, peak_lr, warmup_steps, decay_ratio, min_lr_ratio,
           decay_shape="linear"):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_start = max(warmup_steps, int(total_steps * (1 - decay_ratio)))
    if step < decay_start:
        return peak_lr

    progress = min(max((step - decay_start) / max(1, total_steps - decay_start), 0.0), 1.0)
    if decay_shape == "linear":
        frac = 1.0 - progress
    elif decay_shape == "sqrt":
        frac = 1.0 - math.sqrt(progress)
    elif decay_shape == "cosine":
        frac = 0.5 * (1 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown decay_shape: {decay_shape}")
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * frac)
