import numpy as np
import torch

from hic_unfold.forward import apply_forward, calibrate_d0_tau, soft_contact


def test_soft_contact_numpy_matches_definition():
    D = np.array([[0.0, 100.0, 600.0]])
    out = soft_contact(D, d0=500.0, tau=100.0)
    expected = 1.0 / (1.0 + np.exp((D - 500.0) / 100.0))
    assert np.allclose(out, expected)


def test_soft_contact_torch_matches_numpy():
    D = np.linspace(0, 1000, 50).reshape(5, 10).astype(np.float32)
    out_np = soft_contact(D, d0=500.0, tau=80.0)
    out_t = soft_contact(torch.from_numpy(D), d0=500.0, tau=80.0).numpy()
    assert np.allclose(out_np, out_t, atol=1e-5)


def test_soft_contact_limits():
    D = np.array([0.0, 1e9])
    c = soft_contact(D, d0=500.0, tau=50.0)
    assert c[0] > 0.999  # very close to a contact
    assert c[1] < 1e-9   # very far


def test_apply_forward_returns_ensemble_mean():
    D = np.stack([np.full((4, 4), 100.0), np.full((4, 4), 800.0)])
    H = apply_forward(D, d0=500.0, tau=50.0)
    # cell 0: all entries are contacts (~1); cell 1: none (~0); mean ~ 0.5
    assert np.allclose(H, 0.5, atol=0.01)


def test_apply_forward_is_differentiable_in_torch():
    """The gradient through the forward operator must flow back to D, so guided
    sampling can use it."""
    D = (torch.randn(3, 4, 4).abs() * 200).clone().detach().requires_grad_(True)
    H = apply_forward(D, d0=500.0, tau=80.0)
    loss = H.sum()
    loss.backward()
    assert D.grad is not None
    assert torch.isfinite(D.grad).all()
    assert (D.grad.abs() > 0).any()


def test_calibration_recovers_known_threshold():
    """If H_target IS the soft contact at known (d0*, tau*), calibration
    should recover them (up to grid resolution)."""
    rng = np.random.default_rng(0)
    N = 16; M = 200
    D = rng.uniform(0, 1000, size=(M, N, N))
    D = 0.5 * (D + D.transpose(0, 2, 1))
    for k in range(M):
        np.fill_diagonal(D[k], 0)
    d0_true, tau_true = 450.0, 80.0
    H = apply_forward(D, d0=d0_true, tau=tau_true)
    d0_grid = np.array([350, 400, 450, 500, 550], dtype=np.float32)
    tau_grid = np.array([40, 80, 120, 200], dtype=np.float32)
    res = calibrate_d0_tau(D, H, d0_grid, tau_grid)
    assert res["d0"] == d0_true
    assert res["tau"] == tau_true
    assert res["pearson"] > 0.999
