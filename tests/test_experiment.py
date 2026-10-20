"""Tests of the BOS processing on synthetic data with known displacements."""

import math

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from experiment.bos import compare_truth, process_video
from experiment.design import GAUSS_PEAK, peak_deflection
from experiment.pattern import random_dots
from experiment.processing import CCConfig, displacement, remove_global_motion, replace_outliers
from experiment.synthetic import camera_background, plume_field, warp, write_video
from optics.abel import project


@pytest.fixture(scope="module")
def bg():
    return camera_background(240, 320, seed=2)


def _noisy(img, rng, s=0.01):
    return img + rng.normal(0, s, img.shape)


def test_pattern_fill_fraction():
    img = random_dots(400, 300, dot_px=4.0, fill=0.4, seed=0)
    assert img.shape == (300, 400)
    assert 1 - img.mean() == pytest.approx(0.4, abs=0.03)


@pytest.mark.parametrize("shift", [0.3, 1.7])
def test_uniform_subpixel_shift(bg, shift):
    rng = np.random.default_rng(0)
    cur = _noisy(warp(bg, np.full(bg.shape, shift), np.full(bg.shape, -0.5 * shift)), rng)
    ref = _noisy(bg, rng)
    cc = displacement(ref, cur, "cc")
    assert np.median(cc["dx"]) == pytest.approx(shift, abs=0.01)
    assert np.median(cc["dy"]) == pytest.approx(-0.5 * shift, abs=0.01)
    fl = displacement(ref, cur, "flow")
    s = (slice(16, -16), slice(16, -16))
    assert np.median(fl["dx"][s]) == pytest.approx(shift, abs=0.06)


def test_plume_field_recovered(bg):
    rng = np.random.default_rng(1)
    dx, dy = plume_field(*bg.shape, amp_px=1.0)
    cur, ref = _noisy(warp(bg, dx, dy), rng), _noisy(bg, rng)
    for method, tol in (("cc", 0.03), ("flow", 0.05)):
        d = displacement(ref, cur, method)
        assert compare_truth(d, (dx, dy))["rms_error_px"] < tol, method


def test_outlier_replacement():
    yy, xx = np.mgrid[0:10, 0:12]
    dx, dy = 0.1 * xx.astype(float), 0.05 * yy.astype(float)
    bad_dx, bad_dy = dx.copy(), dy.copy()
    bad_dx[5, 6], bad_dy[5, 6] = 7.0, -4.0
    ndx, ndy, ok = replace_outliers(bad_dx, bad_dy, np.ones_like(dx, bool), CCConfig())
    assert not ok[5, 6]
    assert ndx[5, 6] == pytest.approx(dx[5, 6], abs=0.11) and ndy[5, 6] == pytest.approx(dy[5, 6], abs=0.06)
    assert ok.sum() >= ok.size - 3


def test_global_motion_removed():
    d = {"dx": np.full((5, 5), 0.2), "dy": np.full((5, 5), -0.1)}
    d["dx"][2, 2] = 1.2
    r = remove_global_motion(d)
    assert r["global_shift"] == pytest.approx((0.2, -0.1))
    assert r["dx"][2, 2] == pytest.approx(1.0)


def test_video_pipeline_end_to_end(tmp_path):
    truth = write_video(tmp_path / "rec.mkv", h=192, w=256, n_ref=6, n_flow=8, amp_px=1.0, sway_px=4.0)
    r = process_video(tmp_path / "rec.mkv", slice(0, 6), slice(6, None), "cc")
    assert r["stats"]["n_ref"] == 6 and r["stats"]["n_flow"] == 8
    assert compare_truth(r, truth)["rms_error_px"] < 0.05
    assert r["stats"]["noise_floor_frame_px"] < 0.05


def test_gaussian_index_profile_peak_deflection():
    """For dn(r) = dn0 exp(-r^2/s^2): max |d OPD / dy| = sqrt(2 pi) e^{-1/2} |dn0|, for any s."""
    for s in (0.005, 0.03):
        dr = 4 * s / 800
        r = (np.arange(800) + 0.5) * dr
        y, opd = project(1e-4 * np.exp(-(r / s) ** 2)[:, None], dr)
        eps = np.max(np.abs(np.gradient(opd[:, 0], y)))
        assert eps == pytest.approx(GAUSS_PEAK * 1e-4, rel=0.01)
    # hot plume: width does not matter, deflection grows with temperature excess
    assert peak_deflection(50, 0.005) == pytest.approx(peak_deflection(50, 0.02), rel=0.01)
    assert peak_deflection(100) > peak_deflection(50) > 0
