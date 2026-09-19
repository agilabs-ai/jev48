import torch

from open_system_one.calibration import TemperatureScaler


def ce(logits,targets,t):
    vals=[]
    for z,y in zip(logits,targets): vals.append(-(y*(z/t).log_softmax(-1)).sum())
    return torch.stack(vals).mean().item()


def test_temperature_scaling_improves_soft_target_nll():
    logits=[torch.tensor([5.0,0.0]),torch.tensor([0.0,5.0])]*6
    targets=[torch.tensor([.8,.2]),torch.tensor([.2,.8])]*6
    before=ce(logits,targets,1.0)
    scaler=TemperatureScaler().fit(logits,targets)
    after=ce(logits,targets,scaler)
    assert scaler > 1.0
    assert after < before
