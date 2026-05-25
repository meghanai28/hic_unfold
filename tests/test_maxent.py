import numpy as np

from hic_unfold.maxent import effective_sample_size, maxent_reweight


def test_uniform_weights_have_ess_equal_to_M():
    w = np.ones(50) / 50.0
    assert abs(effective_sample_size(w) - 50.0) < 1e-9


def test_degenerate_weights_have_ess_one():
    w = np.zeros(50); w[3] = 1.0
    assert abs(effective_sample_size(w) - 1.0) < 1e-9


def test_maxent_returns_symmetric_lambda_and_normalised_weights():
    rng = np.random.default_rng(0)
    M, N = 30, 8
    C = rng.uniform(0, 1, size=(M, N, N)).astype(np.float32)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    H = C.mean(axis=0)
    res = maxent_reweight(C, H, num_steps=300, lr=0.1)
    lam = res["lambda"]
    assert np.allclose(lam, lam.T, atol=1e-6)
    assert abs(res["weights"].sum() - 1.0) < 1e-5


def test_maxent_recovers_known_subset():
    """If H is the mean of a specific subset of samples, maxent should put
    weight on those samples and recover H exactly."""
    rng = np.random.default_rng(1)
    M, N = 40, 6
    C = rng.uniform(0, 1, size=(M, N, N)).astype(np.float32)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    subset = rng.choice(M, size=10, replace=False)
    H = C[subset].mean(axis=0)
    res = maxent_reweight(C, H, num_steps=2000, lr=0.05)
    assert res["fit_pearson"] > 0.999, f"Pearson too low: {res['fit_pearson']:.4f}"
    assert res["fit_mse"] < 1e-3, f"MSE too high: {res['fit_mse']:.6f}"
    in_subset_weight = float(res["weights"][subset].sum())
    # Maxent finds the maximum-entropy distribution that matches H — that
    # spreads mass across any samples consistent with the constraint, not
    # necessarily concentrating on the originating subset. Still, the subset
    # mass should be well above uniform (= subset_size / M).
    uniform = len(subset) / M
    assert in_subset_weight > 2 * uniform, (
        f"subset weight {in_subset_weight:.2f} barely above uniform {uniform:.2f}"
    )
