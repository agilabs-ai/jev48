import torch

# Import the script as a module without needing decider installed; the upstream imports are inside main.
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

spec = spec_from_file_location("train_decider_soft", Path("scripts/train_decider_soft.py"))
mod = module_from_spec(spec); spec.loader.exec_module(mod)


def test_soft_loss_prefers_matching_distribution():
    items = [{"target_probs": [0.7, 0.2, 0.1]}]
    good = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).requires_grad_()
    bad = torch.log(torch.tensor([[0.1, 0.2, 0.7]])).requires_grad_()
    lg, _, _ = mod._soft_loss(good, items, 0.2)
    lb, _, _ = mod._soft_loss(bad, items, 0.2)
    assert lg < lb
    lg.backward()
    assert torch.isfinite(good.grad).all()


def test_hard_mode_collapses_only_empirical_targets():
    class E:
        target_probs = [0.55, 0.35, 0.10]
        target_kind = "empirical_distribution"
    assert mod._target_for(E(), [2, 0, 1], "hard") == [0.0, 1.0, 0.0]
    assert mod._target_for(E(), [2, 0, 1], "soft") == [0.10, 0.55, 0.35]
