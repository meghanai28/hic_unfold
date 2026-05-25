import math

import pytest
import torch

from hic_unfold.diffusion import (
    Denoiser,
    ddim_sample,
    make_cosine_schedule,
    q_sample,
    v_to_x0,
)


def test_cosine_schedule_endpoints_and_monotonicity():
    ab = make_cosine_schedule(T=1000)
    assert ab.numel() == 1001
    assert torch.isclose(ab[0], torch.tensor(1.0))
    assert ab[-1] < 0.05
    diffs = ab[1:] - ab[:-1]
    assert (diffs <= 1e-6).all()


def test_q_sample_then_v_to_x0_recovers_x0():
    torch.manual_seed(0)
    ab = make_cosine_schedule(T=500)
    x0 = torch.randn(2, 1, 16, 16)
    t = torch.tensor([100, 300])
    x_t, v = q_sample(x0, t, ab)
    x0_hat = v_to_x0(x_t, v, t, ab)
    assert torch.allclose(x0_hat, x0, atol=1e-5)


def test_denoiser_forward_shape_and_symmetry():
    N, B, d_c = 16, 2, 8
    net = Denoiser(N=N, d_c=d_c, d_h=32, d_t=32, dilations=(1, 2))
    x_t = torch.randn(B, 1, N, N)
    x_t = 0.5 * (x_t + x_t.transpose(-1, -2))
    z = torch.zeros(B, 1, N, N)
    c = torch.randn(B, d_c, N)
    t = torch.tensor([10, 20])
    out = net(x_t, z, c, t)
    assert out.shape == (B, 1, N, N)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)


def test_denoiser_uses_z_conditioning():
    """If z is wired up correctly, perturbing it should change the output."""
    torch.manual_seed(1)
    N, B, d_c = 12, 1, 8
    net = Denoiser(N=N, d_c=d_c, d_h=32, d_t=32, dilations=(1, 2))
    # We must take a tiny gradient step so the zero-init output head produces a
    # non-zero response; otherwise the network always returns zero regardless of z.
    x_t = torch.randn(B, 1, N, N); x_t = 0.5 * (x_t + x_t.transpose(-1, -2))
    z0 = torch.zeros(B, 1, N, N)
    z1 = torch.zeros(B, 1, N, N); z1[:, :, 3, 9] = 1.0; z1[:, :, 9, 3] = 1.0
    c = torch.randn(B, d_c, N)
    t = torch.tensor([100])
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        target = torch.ones_like(x_t)
        out = net(x_t, z0, c, t)
        ((out - target) ** 2).mean().backward()
        opt.step()
    out0 = net(x_t, z0, c, t)
    out1 = net(x_t, z1, c, t)
    assert not torch.allclose(out0, out1, atol=1e-4)


def test_ddim_sampler_runs_and_returns_correct_shape():
    N, B, d_c = 12, 1, 8
    net = Denoiser(N=N, d_c=d_c, d_h=16, d_t=16, dilations=(1,))
    ab = make_cosine_schedule(T=100)
    z = torch.zeros(B, 1, N, N)
    c = torch.randn(B, d_c, N)
    out = ddim_sample(net, z, c, ab, n_steps=8)
    assert out.shape == (B, 1, N, N)
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-5)
