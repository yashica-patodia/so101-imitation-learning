import torch

from nano_vla import config as C
from nano_vla.models.pixel_mae import PATCH, PixelMAE, patchify, unpatchify
from nano_vla.models.probe import ActionProbe


def test_patchify_roundtrip():
    x = torch.rand(1, 2, C.T_IMG, 3, C.FRAME_RES, C.FRAME_RES)
    p = patchify(x)
    assert p.shape == (1, 2, C.T_IMG, (C.FRAME_RES // PATCH) ** 2, 3 * PATCH * PATCH)
    assert torch.allclose(unpatchify(p), x)


def test_pixel_mae_shapes_and_probe():
    B, cams = 2, 2
    n_patches = (C.FRAME_RES // PATCH) ** 2
    frm = torch.rand(B, cams, C.T_IMG, 3, C.FRAME_RES, C.FRAME_RES)
    state = torch.randn(B, C.T_JOINT, C.JOINT_DIM)
    mae = PixelMAE(cams, n_patches, d=64, depth=1, heads=4, dec_d=32, dec_depth=1, dec_heads=4)
    loss, m = mae.loss(frm, state)
    loss.backward()
    assert loss.item() > 0 and m["masked"] > 0
    tokens, _, keep = mae.enc(frm, state)
    assert tokens.shape == (B, cams * C.T_IMG * n_patches + C.T_JOINT, 64) and keep.all()
    out = ActionProbe(64)(tokens, keep, state[:, -1])
    assert out.shape == (B, C.HORIZON, C.JOINT_DIM)
    masked_in, rec, orig = mae.reconstruct(frm, state)
    assert rec.shape == frm.shape and masked_in.shape == frm.shape
