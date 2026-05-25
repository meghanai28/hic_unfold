import numpy as np

from hic_unfold.embedding import classical_mds, mds_residual


def _random_points_distance_matrix(N: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, dim))
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def test_mds_recovers_3d_points_exactly():
    """If D came from 3D points, MDS reconstructs distances exactly."""
    D = _random_points_distance_matrix(N=20, dim=3, seed=0)
    res = mds_residual(D, dim=3)
    assert res["rmse"] < 1e-6
    assert res["relative_rmse"] < 1e-7
    assert res["eig_ratio"] > 0.999


def test_mds_residual_large_for_non_euclidean_matrix():
    """An arbitrary symmetric non-negative matrix won't be Euclidean realisable."""
    rng = np.random.default_rng(1)
    N = 15
    A = rng.uniform(0.5, 1.5, size=(N, N))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    res = mds_residual(A, dim=3)
    assert res["relative_rmse"] > 0.05, "non-Euclidean matrix should have large residual"


def test_classical_mds_returns_correct_shape():
    D = _random_points_distance_matrix(N=12, dim=2, seed=2)
    X, eig = classical_mds(D, dim=3)
    assert X.shape == (12, 3)
    assert eig.shape == (12,)
