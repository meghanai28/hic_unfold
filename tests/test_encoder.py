import torch
import torch.nn.functional as F

from hic_unfold.encoder import LoopEncoder


def test_encoder_forward_shape_and_symmetry():
    N, B, d_c = 16, 2, 8
    net = LoopEncoder(N=N, d_c=d_c, d_h=32, dilations=(1, 2))
    x = torch.randn(B, 1, N, N)
    x = 0.5 * (x + x.transpose(-1, -2))
    c = torch.randn(B, d_c, N)
    out = net(x, c)
    assert out.shape == (B, 1, N, N)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)


def test_encoder_zero_init_starts_at_logit_zero():
    """Zero-initialized output head means untrained predictions are 0 logits."""
    N, B, d_c = 12, 1, 8
    net = LoopEncoder(N=N, d_c=d_c, d_h=16, dilations=(1,))
    x = torch.randn(B, 1, N, N); x = 0.5 * (x + x.transpose(-1, -2))
    c = torch.randn(B, d_c, N)
    out = net(x, c)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_encoder_can_overfit_one_example():
    """Sanity check: with enough capacity and steps, the encoder should be
    able to perfectly fit one (x, z) pair."""
    torch.manual_seed(0)
    N, B, d_c = 16, 1, 8
    net = LoopEncoder(N=N, d_c=d_c, d_h=48, dilations=(1, 2, 4))
    x = torch.randn(B, 1, N, N); x = 0.5 * (x + x.transpose(-1, -2))
    c = torch.randn(B, d_c, N)
    z_true = torch.zeros(B, 1, N, N)
    z_true[:, :, 3, 11] = 1.0; z_true[:, :, 11, 3] = 1.0
    z_true[:, :, 1, 8] = 1.0; z_true[:, :, 8, 1] = 1.0
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    pos_weight = torch.tensor([100.0])
    for _ in range(300):
        opt.zero_grad()
        logits = net(x, c)
        loss = F.binary_cross_entropy_with_logits(logits, z_true, pos_weight=pos_weight)
        loss.backward(); opt.step()
    probs = torch.sigmoid(net(x, c))
    pred = (probs > 0.5).float()
    assert torch.equal(pred, z_true), (
        f"failed to overfit; max prob at true loops: {probs[z_true > 0].min():.3f}, "
        f"max prob elsewhere: {probs[z_true < 1].max():.3f}"
    )
