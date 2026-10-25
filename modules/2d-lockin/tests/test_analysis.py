from __future__ import annotations

import unittest

import numpy as np

from stmfft.analysis import (
    MaskSpec,
    combine_local_amplitudes,
    compute_fft,
    fractional_coordinates,
    local_complex_component,
    make_frequency_mask,
    phase_histogram,
    preprocess_image,
    reconstruct_from_mask,
)


class AnalysisTests(unittest.TestCase):
    def test_composite_amplitude_requires_both_peaks(self) -> None:
        first = np.array([[4.0, 4.0], [0.0, 0.0]])
        second = np.array([[4.0, 0.0], [4.0, 0.0]])
        both = combine_local_amplitudes(first, second, method="geometric", normalize=False)
        self.assertTrue(np.array_equal(both.combined, np.array([[4.0, 0.0], [0.0, 0.0]])))
        total = combine_local_amplitudes(first, second, method="rms", normalize=False)
        self.assertGreater(float(total.combined[0, 1]), 0.0)
        weaker = combine_local_amplitudes(first, second, method="minimum", normalize=False)
        self.assertTrue(np.array_equal(weaker.combined, both.combined))

    def test_composite_means_have_clear_order_and_zero_rule(self) -> None:
        first = np.array([[4.0, 0.0]])
        second = np.array([[1.0, 0.0]])
        harmonic = combine_local_amplitudes(first, second, normalize=False).combined
        geometric = combine_local_amplitudes(first, second, method="geometric", normalize=False).combined
        quadratic = combine_local_amplitudes(first, second, method="rms", normalize=False).combined
        minimum = combine_local_amplitudes(first, second, method="minimum", normalize=False).combined
        self.assertAlmostEqual(float(harmonic[0, 0]), 1.6)
        self.assertAlmostEqual(float(geometric[0, 0]), 2.0)
        self.assertAlmostEqual(float(quadratic[0, 0]), np.sqrt(8.5))
        self.assertLess(float(minimum[0, 0]), float(harmonic[0, 0]))
        self.assertLess(float(harmonic[0, 0]), float(geometric[0, 0]))
        self.assertLess(float(geometric[0, 0]), float(quadratic[0, 0]))
        self.assertEqual(float(harmonic[0, 1]), 0.0)

    def test_composite_p90_normalization_removes_global_peak_scale(self) -> None:
        first = np.arange(1.0, 17.0).reshape(4, 4)
        second = first * 12.0
        result = combine_local_amplitudes(first, second, normalize=True)
        self.assertTrue(np.allclose(result.first, result.second))
        self.assertTrue(np.allclose(result.combined, result.first))
        self.assertAlmostEqual(result.second_scale / result.first_scale, 12.0)

    def test_composite_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            combine_local_amplitudes(np.ones((2, 2)), np.ones((2, 3)))

    def test_plane_subtraction(self) -> None:
        yy, xx = np.mgrid[0:80, 0:96]
        image = 2.3 * xx - 0.7 * yy + 4.0
        flattened, fraction = preprocess_image(image, "plane")
        self.assertAlmostEqual(fraction, 1.0)
        self.assertLess(float(np.std(flattened)), 1e-10)

    def test_fft_peak_and_pair_reconstruction(self) -> None:
        nx = ny = 128
        width = height = 16.0
        x = np.arange(nx) * width / nx
        image = np.cos(2 * np.pi * 1.25 * x)[None, :] * np.ones((ny, 1))
        result = compute_fft(image, width, height, window="none")
        spec = MaskSpec(qx=1.25, qy=0.0, radius=0.09, shape="hard", include_conjugate=True)
        mask = make_frequency_mask(result.qx, result.qy, [spec])
        reconstructed = reconstruct_from_mask(result.spectrum, mask)
        correlation = np.corrcoef(image.ravel(), reconstructed.ravel())[0, 1]
        self.assertGreater(correlation, 0.999)

    def test_single_side_amplitude_and_phase(self) -> None:
        nx = ny = 128
        width = height = 16.0
        amplitude = 2.4
        phase = 0.63
        x = np.arange(nx) * width / nx
        image = amplitude * np.cos(2 * np.pi * 1.25 * x + phase)[None, :] * np.ones((ny, 1))
        result = compute_fft(image, width, height, window="none")
        spec = MaskSpec(qx=1.25, qy=0.0, radius=0.09, shape="hard", include_conjugate=False)
        component = local_complex_component(result.spectrum, result.qx, result.qy, spec)
        self.assertAlmostEqual(float(np.mean(np.abs(component))), amplitude, places=8)
        measured = np.angle(np.mean(component))
        self.assertAlmostEqual(float(measured), phase, places=7)

    def test_fractional_bragg_coordinates(self) -> None:
        g1 = (2.6, 0.1)
        g2 = (-0.1, 2.5)
        q = (0.75 * g1[0] + 0.25 * g2[0], 0.75 * g1[1] + 0.25 * g2[1])
        h, k = fractional_coordinates(q, g1, g2)
        self.assertAlmostEqual(h, 0.75)
        self.assertAlmostEqual(k, 0.25)

    def test_hann_window_creates_central_spatial_envelope(self) -> None:
        nx = ny = 128
        width = height = 16.0
        x = np.arange(nx) * width / nx
        image = np.cos(2 * np.pi * 1.25 * x)[None, :] * np.ones((ny, 1))
        spec = MaskSpec(qx=1.25, qy=0.0, radius=0.3, shape="hard", include_conjugate=False)

        raw_fft = compute_fft(image, width, height, window="none")
        raw_amp = np.abs(local_complex_component(raw_fft.spectrum, raw_fft.qx, raw_fft.qy, spec))
        self.assertLess(float(np.std(raw_amp) / np.mean(raw_amp)), 1e-10)

        hann_fft = compute_fft(image, width, height, window="hann")
        hann_amp = np.abs(local_complex_component(hann_fft.spectrum, hann_fft.qx, hann_fft.qy, spec))
        center = float(np.mean(hann_amp[48:80, 48:80]))
        edges = float(np.mean(np.concatenate((hann_amp[:12].ravel(), hann_amp[-12:].ravel()))))
        self.assertGreater(center, 20.0 * edges)

    def test_phase_histogram_uses_circular_mean(self) -> None:
        phases = np.array([np.pi - 0.08, -np.pi + 0.08, np.pi - 0.04, -np.pi + 0.04])
        component = np.exp(1j * phases).reshape(2, 2)
        result = phase_histogram(component, amplitude_floor_percentile=0, bins=24)
        self.assertEqual(result["pixel_count"], 4)
        self.assertGreater(result["resultant_length"], 0.99)
        self.assertLess(abs(abs(result["mean_phase"]) - np.pi), 1e-8)

    def test_phase_histogram_applies_amplitude_floor(self) -> None:
        component = np.array([1.0, 2.0, 3.0, 4.0]) * np.exp(1j * 0.4)
        result = phase_histogram(component, amplitude_floor_percentile=50, bins=12)
        self.assertEqual(result["pixel_count"], 2)
        self.assertAlmostEqual(result["mean_phase"], 0.4, places=12)


if __name__ == "__main__":
    unittest.main()
