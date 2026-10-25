from __future__ import annotations

import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from stmfft.gui import STMFFTApp, _application_dir, _orient_array
from stmfft.palette import make_lut


class GuiHelperTests(unittest.TestCase):
    def test_application_dir_uses_exe_location_when_frozen(self) -> None:
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", r"C:\Portable\STM_Fourier_Analyzer\STM_Fourier_Analyzer.exe"):
                self.assertEqual(_application_dir(), Path(sys.executable).resolve().parent)

    def test_reciprocal_display_flips_both_axes(self) -> None:
        array = np.array([[1, 2, 3], [4, 5, 6]])
        shown = _orient_array(array, flip_x=True, flip_y=True)
        np.testing.assert_array_equal(shown, np.array([[6, 5, 4], [3, 2, 1]]))

    def test_phase_histogram_color_moves_with_bin(self) -> None:
        bins = 8
        values = np.zeros(bins)
        values[0] = 10
        edges = np.linspace(-np.pi, np.pi, bins + 1)
        result = {
            "histogram": values,
            "edges": edges,
            "amplitude_weighted": False,
        }
        app = object.__new__(STMFFTApp)
        original = app._phase_histogram_image(result, 0.0)
        shifted = app._phase_histogram_image(result, np.pi / 2)
        expected = tuple(int(channel) for channel in make_lut("phase")[16])
        # First-bin center moves by one quarter of the plot width, carrying
        # its original blue color instead of being recolored by x position.
        original_pixel = original.getpixel((72 + 20, 350))
        shifted_pixel = shifted.getpixel((72 + (748 - 72) // 4 + 20, 350))
        self.assertEqual(original_pixel, expected)
        self.assertEqual(shifted_pixel, expected)

    def test_export_colorbar_margin_is_transparent(self) -> None:
        app = object.__new__(STMFFTApp)
        source = Image.new("RGB", (120, 80), "red")
        panel = SimpleNamespace(
            palette="phase",
            colorbar_ticks=("+π", "0", "−π"),
            colorbar_label="相位 (rad)",
        )
        exported = app._append_export_colorbar(source, panel)
        self.assertEqual(exported.mode, "RGBA")
        self.assertEqual(exported.getpixel((exported.width - 1, exported.height - 1))[3], 0)
        self.assertEqual(exported.getpixel((10, 10))[3], 255)

    def test_safe_filename_part(self) -> None:
        self.assertEqual(STMFFTApp._safe_filename_part('Q1:a/b?c*'), "Q1_a_b_c_")


if __name__ == "__main__":
    unittest.main()
