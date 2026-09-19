from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import torch

from .batching import collate_requests
from .model import load_hf_backbone, load_head_checkpoint
from .schema import (
    Candidate, ChoiceAnswer, DecisionRequest, DecisionResponse,
    SystemOneRequest, SystemOneResponse,
)


@dataclass
class LoadedEngine:
    model: object
    tokenizer: object
    device: str
    max_length: int
    temperature: float

    @torch.no_grad()
    def _predict_many(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        batch = collate_requests(requests, self.tokenizer, self.max_length).to(self.device)
        out = self.model(batch.input_ids, batch.attention_mask, batch.group_sizes)
        results=[]
        for i,(request,k) in enumerate(zip(requests,batch.group_sizes)):
            logits = out.logits[i, :k].float() / self.temperature
            probs = logits.softmax(-1).cpu().tolist()
            best = max(range(k), key=probs.__getitem__)
            results.append(DecisionResponse(
                best=request.candidates[best].id,
                probabilities={c.id: float(p) for c, p in zip(request.candidates, probs)},
                temperature=self.temperature,
                metadata={"output_tokens": 0, "candidate_paths": k, "prefix_sharing": False},
            ))
        return results

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        return self._predict_many([request])[0]

    def system_one(self, request: SystemOneRequest) -> SystemOneResponse:
        qids=list(request.questions)
        decisions=[]
        for qid in qids:
            q=request.questions[qid]
            decisions.append(DecisionRequest(
                state=request.state,
                question=q.instructions,
                candidates=[Candidate(id=k,text=v) for k,v in q.criteria.items()],
            ))
        predictions=self._predict_many(decisions)
        answers={}
        total_paths=0
        for qid,pred in zip(qids,predictions):
            confidence=max(pred.probabilities.values())
            answers[qid]=ChoiceAnswer(choice=pred.best,probabilities=pred.probabilities,confidence=confidence)
            total_paths+=len(pred.probabilities)
        return SystemOneResponse(
            model="open-system-one",
            answers=answers,
            usage={"output_tokens":0,"candidate_paths":total_paths,"questions":len(qids)},
            metadata={"prefix_sharing":False,"temperature":self.temperature,"confidence_semantics":"max calibrated choice probability"},
        )


def load_engine(checkpoint: str | Path, device: str | None = None) -> LoadedEngine:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("pip install -e '.[train,serve]'") from exc
    p = Path(checkpoint)
    metadata = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(metadata["base_model"], revision=metadata.get("revision"))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = load_hf_backbone(
        metadata["base_model"], device=device, dtype=metadata.get("inference_dtype", "auto"), revision=metadata.get("revision")
    )
    if metadata.get("train_mode") == "lora":
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("LoRA checkpoint requires peft: pip install -e '.[train]'") from exc
        backbone = PeftModel.from_pretrained(backbone, p / metadata.get("adapter_path", "adapter"))
    model, metadata = load_head_checkpoint(p, backbone)
    model = model.to(device).eval()
    return LoadedEngine(model, tokenizer, device, int(metadata.get("max_length", 2048)), float(metadata.get("temperature", 1.0)))
