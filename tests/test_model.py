from types import SimpleNamespace
import torch
from torch import nn

from openjev.model import DynamicDecisionModel,soft_target_loss


class FakeBackbone(nn.Module):
    def __init__(self,hidden=16,vocab=100):
        super().__init__(); self.config=SimpleNamespace(hidden_size=hidden); self.emb=nn.Embedding(vocab,hidden)
    def forward(self,input_ids,attention_mask,use_cache=False):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def test_dynamic_shapes_and_loss():
    torch.manual_seed(0)
    model=DynamicDecisionModel(FakeBackbone(),set_attention=True,set_dim=16)
    ids=torch.randint(0,100,(5,7)); mask=torch.ones_like(ids)
    out=model(ids,mask,[2,3])
    assert out.logits.shape==(2,3)
    assert out.valid_mask.tolist()==[[True,True,False],[True,True,True]]
    targets=torch.tensor([[.2,.8,0],[.1,.2,.7]],dtype=torch.float32)
    loss,parts=soft_target_loss(out.logits,out.valid_mask,targets)
    assert torch.isfinite(loss)
    assert parts["brier"] >= 0


def test_set_head_is_permutation_equivariant():
    torch.manual_seed(0)
    model=DynamicDecisionModel(FakeBackbone(),set_attention=True,set_dim=16).eval()
    a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    mask=torch.ones_like(a)
    z1=model(a,mask,[3]).logits[0,:3]
    order=torch.tensor([2,0,1])
    z2=model(a[order],mask[order],[3]).logits[0,:3]
    assert torch.allclose(z2,z1[order],atol=1e-6)
