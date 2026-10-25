from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from stmfft.sxm import read_sxm


class SxmReaderTests(unittest.TestCase):
    @unittest.skipUnless(len(sys.argv) > 1 and Path(sys.argv[-1]).suffix.lower() == ".sxm", "可选的真实 SXM 路径未提供")
    def test_real_sxm(self) -> None:
        sxm = read_sxm(sys.argv[-1])
        self.assertGreaterEqual(len(sxm.channels), 1)
        self.assertGreater(sxm.scan_pixels[0], 1)
        self.assertGreater(sxm.scan_pixels[1], 1)
        self.assertTrue(np.any(np.isfinite(sxm.channels[0].data)))


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:-1])
