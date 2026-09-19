from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Sequence

import torch
from torch import nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    valid_mask: torch.Tensor


class DynamicDecisionModel(nn.Module):
    """Variable-cardinality decision head over a transformer encoder/backbone.

    V0 intentionally batches *complete candidate paths* in one backbone forward.
    This is zero autoregressive decoding, but it is not prefix sharing. The
    benchmark suite measures candidate-count scaling so that this limitation is
    visible rather than hidden.
    """

    def __init__(self, backbone: nn.Module, set_attention: bool = True, set_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        hidden = int(backbone.config.hidden_size)
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.use_set_attention = bool(set_attention)
        self.set_dim = int(set_dim)
        if self.use_set_attention:
            self.set_project = nn.Linear(hidden + 1, set_dim)
            self.set_attention = nn.MultiheadAttention(set_dim, 4, batch_first=True, dropout=0.0)
            self.set_output = nn.Linear(set_dim, 1)
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, group_sizes: Sequence[int]) -> ModelOutput:
        if sum(group_sizes) != input_ids.shape[0]:
            raise ValueError("sum(group_sizes) must equal flattened candidate path batch size")
        if min(group_sizes) < 2:
            raise ValueError("each question must contain at least two candidates")
        body = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = body.last_hidden_state
        lengths = attention_mask.long().sum(-1).clamp_min(1)
        leaf = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]

        batch = len(group_sizes)
        kmax = max(group_sizes)
        h = leaf.new_zeros((batch, kmax, leaf.shape[-1]))
        valid = torch.zeros((batch, kmax), dtype=torch.bool, device=leaf.device)
        offset = 0
        for i, k in enumerate(group_sizes):
            h[i, :k] = leaf[offset:offset + k]
            valid[i, :k] = True
            offset += k

        h = self.norm(h)
        logits = self.scalar(h).squeeze(-1).float()
        if self.use_set_attention:
            log_k = valid.sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
            u = self.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
            logits = logits + self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        logits = logits.masked_fill(~valid, -1e9)
        return ModelOutput(logits=logits, valid_mask=valid)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        missing = []
        for key, value in state.items():
            if key not in own:
                raise KeyError(f"unknown head key: {key}")
            own[key].copy_(value.to(dtype=own[key].dtype, device=own[key].device))
        for key in own:
            if not key.startswith("backbone.") and key not in state:
                missing.append(key)
        if missing:
            raise KeyError(f"missing head keys: {missing}")


def soft_target_loss(logits: torch.Tensor, valid_mask: torch.Tensor, targets: torch.Tensor,
                     ce_weight: float = 1.0, brier_weight: float = 0.25) -> tuple[torch.Tensor, dict[str, float]]:
    logp = logits.log_softmax(-1)
    probs = logits.softmax(-1)
    ce = -(targets * logp).sum(-1).mean()
    brier = ((probs - targets) ** 2 * valid_mask.float()).sum(-1).mean()
    loss = ce_weight * ce + brier_weight * brier
    return loss, {"loss": float(loss.detach()), "ce": float(ce.detach()), "brier": float(brier.detach())}


def load_hf_backbone(model_name: str, *, device: str, dtype: str = "auto", revision: str | None = None):
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from exc
    kwargs = {"trust_remote_code": False}
    if revision:
        kwargs["revision"] = revision
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    backbone = AutoModel.from_pretrained(model_name, **kwargs)
    return backbone.to(device)


def save_head_checkpoint(path: str | Path, model: DynamicDecisionModel, metadata: dict) -> None:
    from safetensors.torch import save_file
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    save_file(model.head_state_dict(), str(p / "decision_head.safetensors"))
    (p / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_head_checkpoint(path: str | Path, backbone: nn.Module) -> tuple[DynamicDecisionModel, dict]:
    from safetensors.torch import load_file
    p = Path(path)
    metadata = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    model = DynamicDecisionModel(
        backbone,
        set_attention=metadata.get("set_attention", True),
        set_dim=int(metadata.get("set_dim", 128)),
    )
    model.load_head_state_dict(load_file(str(p / "decision_head.safetensors")))
    return model, metadata
