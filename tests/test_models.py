import torch

from nano_vla import config as C
from nano_vla.models.mae import MAE, sample_mask
from nano_vla.models.probe import ActionProbe


def test_mae_and_probe_shapes():
    B, cams, P, D = 2, 2, C.DINO_N_PATCHES, C.DINO_FEAT_DIM
    img = torch.randn(B, cams, C.T_IMG, P, D)
    state = torch.randn(B, C.T_JOINT, C.JOINT_DIM)
    mae = MAE(cams, P, D)
    loss, m = mae.loss(img, state)
    loss.backward()
    assert loss.item() > 0 and "masked" in m
    tokens, _, keep = mae.enc(img, state)
    assert tokens.shape == (B, cams * C.T_IMG * P + C.T_JOINT, C.D_MODEL)
    out = ActionProbe()(tokens, keep, state[:, -1])
    assert out.shape == (B, C.HORIZON, C.JOINT_DIM)


def test_mask_keep_rate_is_reasonable():
    torch.manual_seed(0)
    k = sample_mask(64, 2, C.DINO_N_PATCHES, C.T_JOINT, "cpu")
    frac = k.float().mean().item()
    assert 0.15 < frac < 0.5
