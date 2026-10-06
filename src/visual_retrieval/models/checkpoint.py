"""Restricted PyTorch checkpoint deserialization."""
import torch


def load_checkpoint(path):
    # Stage2 checkpoints can contain built-in set metadata alongside tensors.
    # Keep weights_only=True and scope this one extra type to the load. Avoid
    # removing a set allowlist entry that was already provided by the caller.
    extra = [] if set in torch.serialization.get_safe_globals() else [set]
    with torch.serialization.safe_globals(extra):
        return torch.load(path, map_location='cpu', weights_only=True)
