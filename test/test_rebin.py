import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.lib.rebin import rebin2d
from test.specula_testlib import cpu_and_gpu


class TestRebin(unittest.TestCase):

    @cpu_and_gpu
    def test_rebin2d_rectangular(self, target_device_idx, xp):
        '''Downsampling of rectangular arrays, also with one axis unchanged'''
        a = np.arange(24 * 12, dtype=np.float64).reshape(24, 12)
        for shape in [(6, 3), (24, 3), (6, 12), (8, 12), (12, 6)]:
            with self.subTest(shape=shape):
                m, n = shape
                expected = a.reshape(m, 24 // m, n, 12 // n).mean(axis=(1, 3))
                out = rebin2d(xp.asarray(a), shape, xp=xp)
                np.testing.assert_allclose(cpuArray(out), expected)
                out_t = rebin2d(xp.asarray(a.T), (n, m), xp=xp)
                np.testing.assert_allclose(cpuArray(out_t), expected.T)

    @cpu_and_gpu
    def test_rebin2d_errors(self, target_device_idx, xp):
        a = xp.zeros((24, 12))
        with self.assertRaises(ValueError):
            rebin2d(a, (48, 24), xp=xp)          # upsampling without sample=True
        with self.assertRaises(ValueError):
            rebin2d(a, (12, 24), xp=xp)          # downsampling and upsampling
        with self.assertRaises(ValueError):
            rebin2d(a, (5, 12), xp=xp)           # non-integer factor
