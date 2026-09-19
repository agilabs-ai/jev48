from __future__ import annotations

import torch

from .batching import collate_examples
from .metrics import summarize


@torch.no_grad()
def predict_examples(model, tokenizer, examples, *, device: str, max_length: int, batch_questions: int = 8, temperature: float = 1.0):
    model.eval()
    probs=[]; targets=[]; logits_out=[]; records=[]
    for start in range(0,len(examples),batch_questions):
        group=examples[start:start+batch_questions]
        batch=collate_examples(group,tokenizer,max_length,permute_candidates=False).to(device)
        out=model(batch.input_ids,batch.attention_mask,batch.group_sizes)
        for i,(ex,k) in enumerate(zip(group,batch.group_sizes)):
            z=out.logits[i,:k].float().cpu()
            p=(z/temperature).softmax(-1)
            probs.append(p.tolist())
            targets.append(ex.target_probs)
            logits_out.append(z)
            records.append({
                "id":ex.id,"split":ex.split,"domain":ex.domain,
                "candidate_ids":[c.id for c in ex.candidates],
                "target_probs":ex.target_probs,"target_kind":ex.target_kind,
                "logits":z.tolist(),"probabilities":p.tolist(),
            })
    return summarize(probs,targets), logits_out, targets, records
