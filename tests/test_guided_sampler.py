import numpy as np
import torch

from hic_unfold.diffusion import (
    Denoiser,
    guided_ddim_sample,
    make_cosine_schedule,
)


def test_guided_sampler_runs_and_returns_correct_shapes():
    N, M, d_c = 12, 3, 8
    net = Denoiser(N=N, d_c=d_c, d_h=16, d_t=16, dilations=(1,))
    ab = make_cosine_schedule(T=50)
    z = torch.zeros(M, 1, N, N)
    c = torch.randn(M, d_c, N)
    H_obs = torch.rand(N, N) * 0.5
    H_obs = 0.5 * (H_obs + H_obs.T)

    res = guided_ddim_sample(net, z, c, ab, H_obs,
                              d0=500.0, tau=80.0, mu=6.0, sigma=1.0,
                              n_steps=5, eta=1.0)
    assert res["D"].shape == (M, N, N)
    assert torch.allclose(res["D"], res["D"].transpose(-1, -2), atol=1e-5)
    diag = res["D"][:, torch.arange(N), torch.arange(N)]
    assert (diag.abs() < 1e-6).all()
    assert len(res["losses"]) == 5
    assert all(np.isfinite(res["losses"]))


def test_guided_sampler_reduces_loss():
    """Even with a randomly-initialised denoiser, the guidance gradient should
    push the ensemble bulk closer to H_obs over many steps."""
    torch.manual_seed(0)
    N, M, d_c = 16, 8, 8
    net = Denoiser(N=N, d_c=d_c, d_h=32, d_t=32, dilations=(1, 2))
    # warm up output head so guidance has signal to work with
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(10):
        x = torch.randn(M, 1, N, N); x = 0.5 * (x + x.transpose(-1, -2))
        z = torch.zeros(M, 1, N, N)
        c = torch.randn(M, d_c, N)
        t = torch.randint(1, 50, (M,))
        target = torch.randn_like(x); target = 0.5 * (target + target.transpose(-1, -2))
        opt.zero_grad()
        loss = ((net(x, z, c, t) - target) ** 2).mean()
        loss.backward(); opt.step()

    ab = make_cosine_schedule(T=100)
    z = torch.zeros(M, 1, N, N)
    c = torch.randn(M, d_c, N)
    H_obs = torch.full((N, N), 0.3)

    res = guided_ddim_sample(net, z, c, ab, H_obs,
                              d0=500.0, tau=80.0, mu=6.0, sigma=1.0,
                              n_steps=40, eta=2.0)
    early = np.mean(res["losses"][:5])
    late = np.mean(res["losses"][-5:])
    assert late < early, f"loss should decrease: early={early:.6f}, late={late:.6f}"


def test_guidance_gradient_is_finite_and_nonzero():
    """The gradient through the forward operator must be finite and not vanish."""
    torch.manual_seed(0)
    N, M, d_c = 12, 4, 8
    net = Denoiser(N=N, d_c=d_c, d_h=16, d_t=16, dilations=(1,))
    ab = make_cosine_schedule(T=20)
    z = torch.zeros(M, 1, N, N)
    c = torch.randn(M, d_c, N)
    H_obs = torch.full((N, N), 0.5)

    res = guided_ddim_sample(net, z, c, ab, H_obs,
                              d0=500.0, tau=80.0, mu=6.0, sigma=1.0,
                              n_steps=5, eta=1.0)
    gn = np.array(res["grad_norms"])
    assert np.all(np.isfinite(gn))
    assert (gn > 0).any()
