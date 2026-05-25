import numpy as np
import pytest

from hic_unfold.simulator.loop_extrusion import (
    LoopExtrusionConfig,
    run_to_snapshot,
    simulate_ensemble,
    snapshot_to_loop_matrix,
)


def _zero_ctcf(N: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(N), np.zeros(N)


def test_config_rejects_bad_shapes():
    N = 20
    with pytest.raises(ValueError):
        LoopExtrusionConfig(
            N=N, num_lefs=2, processivity=10.0,
            ctcf_left_stop=np.zeros(N - 1), ctcf_right_stop=np.zeros(N),
        )


def test_config_rejects_bad_probabilities():
    N = 20
    with pytest.raises(ValueError):
        LoopExtrusionConfig(
            N=N, num_lefs=2, processivity=10.0,
            ctcf_left_stop=np.full(N, 1.5), ctcf_right_stop=np.zeros(N),
        )


def test_legs_stay_in_bounds():
    N = 40
    cl, cr = _zero_ctcf(N)
    cfg = LoopExtrusionConfig(N=N, num_lefs=4, processivity=200.0,
                              ctcf_left_stop=cl, ctcf_right_stop=cr)
    rng = np.random.default_rng(0)
    L, R = run_to_snapshot(cfg, num_steps=500, rng=rng)
    assert (L >= 0).all() and (L < N).all()
    assert (R >= 0).all() and (R < N).all()
    assert (L <= R).all()


def test_no_two_legs_share_a_site():
    N = 40
    cl, cr = _zero_ctcf(N)
    cfg = LoopExtrusionConfig(N=N, num_lefs=6, processivity=200.0,
                              ctcf_left_stop=cl, ctcf_right_stop=cr)
    rng = np.random.default_rng(1)
    L, R = run_to_snapshot(cfg, num_steps=300, rng=rng)
    sites = []
    for li, ri in zip(L.tolist(), R.tolist()):
        sites.append(li)
        if ri != li:
            sites.append(ri)
    assert len(sites) == len(set(sites)), "two LEF legs share a site"


def test_lefs_extrude_without_barriers():
    """With no CTCF and high processivity, LEFs should grow well beyond their load site."""
    N = 60
    cl, cr = _zero_ctcf(N)
    cfg = LoopExtrusionConfig(N=N, num_lefs=2, processivity=10_000.0,
                              ctcf_left_stop=cl, ctcf_right_stop=cr)
    rng = np.random.default_rng(2)
    L, R = run_to_snapshot(cfg, num_steps=400, rng=rng)
    spans = (R - L)
    assert spans.mean() > 10, f"LEFs failed to extrude (mean span={spans.mean()})"


def test_convergent_ctcf_produces_corner_peak():
    """Headline verification: with one LEF and perfect convergent CTCFs at (a,b),
    the averaged loop matrix is dominated by the (a,b) entry — LEFs loaded in
    [a,b] get trapped there, while LEFs loaded outside stall at one barrier and
    one boundary."""
    N = 50
    a, b = 15, 35
    cl = np.zeros(N)
    cr = np.zeros(N)
    cl[a] = 1.0  # forward CTCF at a — perfectly stalls left-moving legs
    cr[b] = 1.0  # reverse CTCF at b — perfectly stalls right-moving legs

    cfg = LoopExtrusionConfig(
        N=N, num_lefs=1, processivity=500.0,
        ctcf_left_stop=cl, ctcf_right_stop=cr,
    )
    result = simulate_ensemble(cfg, num_cells=400, steps_per_cell=400, seed=7)
    Z = result.avg_loop_matrix

    corner = Z[a, b]
    # Mask the diagonal and (a,b)/(b,a); the corner should dominate the rest.
    other = Z.copy()
    other[a, b] = 0
    other[b, a] = 0
    np.fill_diagonal(other, 0)

    assert corner > 0.1, f"corner peak too weak: Z[{a},{b}]={corner:.4f}"
    assert corner > other.max() * 1.2, (
        f"corner peak Z[{a},{b}]={corner:.4f} not dominant over next entry {other.max():.4f}"
    )


def test_loop_matrix_is_symmetric_with_zero_diagonal():
    N = 30
    cl, cr = _zero_ctcf(N)
    cfg = LoopExtrusionConfig(N=N, num_lefs=3, processivity=100.0,
                              ctcf_left_stop=cl, ctcf_right_stop=cr)
    rng = np.random.default_rng(3)
    L, R = run_to_snapshot(cfg, num_steps=200, rng=rng)
    z = snapshot_to_loop_matrix(L, R, N)
    assert np.array_equal(z, z.T)
    assert np.diag(z).sum() == 0
