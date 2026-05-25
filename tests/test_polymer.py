import numpy as np
import pytest

from hic_unfold.polymer.gaussian import (
    PolymerConfig,
    build_laplacian,
    sample_distance_matrix,
    sample_positions,
    soft_contact,
)


def _no_loops(N: int) -> np.ndarray:
    return np.zeros((N, N), dtype=np.int8)


def test_config_rejects_bad_strengths():
    with pytest.raises(ValueError):
        PolymerConfig(backbone_k=0)
    with pytest.raises(ValueError):
        PolymerConfig(loop_k=-1.0)


def test_laplacian_row_sums_are_zero():
    """A graph Laplacian always has zero row sums."""
    N = 30
    cfg = PolymerConfig()
    K = build_laplacian(_no_loops(N), N, cfg)
    assert np.allclose(K.sum(axis=1), 0.0)


def test_laplacian_is_symmetric_and_psd():
    N = 30
    z = _no_loops(N)
    z[5, 20] = 1
    z[20, 5] = 1
    K = build_laplacian(z, N, PolymerConfig())
    assert np.allclose(K, K.T)
    eig = np.linalg.eigvalsh(K)
    assert eig.min() > -1e-10  # PSD (smallest eigenvalue should be ~0, others > 0)


def test_distance_matrix_symmetric_zero_diagonal():
    N = 40
    rng = np.random.default_rng(0)
    D, _ = sample_distance_matrix(_no_loops(N), N, rng)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)


def test_distance_matrix_satisfies_triangle_inequality():
    """Real Euclidean distances obey d(i,k) <= d(i,j) + d(j,k) for all i,j,k."""
    N = 25
    rng = np.random.default_rng(1)
    D, _ = sample_distance_matrix(_no_loops(N), N, rng)
    for _ in range(20):
        i, j, k = rng.integers(0, N, size=3)
        assert D[i, k] <= D[i, j] + D[j, k] + 1e-9


def test_free_chain_distance_scales_with_sqrt_separation():
    """For a Gaussian chain with no loops, <d^2(i,j)> grows linearly in |i-j|."""
    N = 80
    rng = np.random.default_rng(2)
    z = _no_loops(N)
    n_samples = 200
    d2_sum = np.zeros((N, N))
    for _ in range(n_samples):
        D, _ = sample_distance_matrix(z, N, rng)
        d2_sum += D ** 2
    d2_mean = d2_sum / n_samples

    seps = [5, 10, 20, 40]
    d2_by_sep = []
    for s in seps:
        diag = np.diag(d2_mean, k=s)
        d2_by_sep.append(diag.mean())

    # Linear fit: d2 vs separation should be roughly proportional (intercept ~0)
    ratios = [d2_by_sep[i] / seps[i] for i in range(len(seps))]
    assert max(ratios) / min(ratios) < 1.5, f"non-linear scaling: ratios={ratios}"


def test_loop_anchor_pulls_pair_closer():
    """Adding a loop spring between i, j should reduce <d(i,j)>."""
    N = 60
    i, j = 5, 55
    rng = np.random.default_rng(3)

    z_no = _no_loops(N)
    z_yes = _no_loops(N)
    z_yes[i, j] = 1
    z_yes[j, i] = 1

    n_samples = 200
    d_no = 0.0
    d_yes = 0.0
    for _ in range(n_samples):
        D_no, _ = sample_distance_matrix(z_no, N, rng)
        D_yes, _ = sample_distance_matrix(z_yes, N, rng)
        d_no += D_no[i, j]
        d_yes += D_yes[i, j]
    d_no /= n_samples
    d_yes /= n_samples
    assert d_yes < 0.5 * d_no, (
        f"loop did not pull anchors close enough: no_loop={d_no:.3f}, with_loop={d_yes:.3f}"
    )


def test_soft_contact_monotone_decreasing():
    D = np.array([[0.0, 1.0, 5.0],
                  [1.0, 0.0, 3.0],
                  [5.0, 3.0, 0.0]])
    C = soft_contact(D, d0=4.0, tau=1.0)
    assert C[0, 0] > C[0, 1] > C[1, 2] > C[0, 2]
    assert ((C >= 0) & (C <= 1)).all()


def test_sample_positions_centre_of_mass_at_origin():
    """Skipping the zero eigenvalue fixes the centre of mass at the origin."""
    N = 50
    cfg = PolymerConfig()
    K = build_laplacian(_no_loops(N), N, cfg)
    rng = np.random.default_rng(4)
    X = sample_positions(K, rng)
    com = X.mean(axis=0)
    assert np.allclose(com, 0.0, atol=1e-9)
