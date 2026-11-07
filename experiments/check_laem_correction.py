"""Small data-free check against the authors' adjusted pre-update prediction rule."""
import copy
import torch
import cmn as C

torch.manual_seed(0)
s=C.Source(12,4,h=16).eval().requires_grad_(False)
x=torch.randn(2,7,12)
expected=[]
for batch in x:
    native=copy.deepcopy(s)
    native.bn.train()
    z=native(batch)
    counts=torch.bincount(z.argmax(-1),minlength=4).float().clamp_min(1e-12)
    prior=counts.sqrt();prior/=prior.sum()
    expected.append((z+prior.log()).softmax(-1)[0])
p,_,record=C.run_laem(s,x,steps=1,record=[1])
torch.testing.assert_close(p,torch.stack(expected),atol=1e-6,rtol=1e-5)
torch.testing.assert_close(p,record[1])
p,_,record=C.run_laem(s,x,steps=3,record=[1,3])
assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(2))
torch.testing.assert_close(p,record[3])
print('LAEM official-sign/prior/readout and finite-step checks passed')
