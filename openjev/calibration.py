from __future__ import annotations

import math
import torch


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: list[torch.Tensor], targets: list[torch.Tensor], max_iter: int = 100) -> float:
        if len(logits) != len(targets) or not logits:
            raise ValueError("logits and targets must be nonempty and aligned")
        log_t = torch.tensor(math.log(self.temperature), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            t = log_t.exp().clamp(1e-3, 1e3)
            losses = []
            for z, y in zip(logits, targets):
                z64 = z.detach().double() / t
                y64 = y.detach().double()
                losses.append(-(y64 * z64.log_softmax(-1)).sum())
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.detach().exp().clamp(1e-3, 1e3))
        return self.temperature
