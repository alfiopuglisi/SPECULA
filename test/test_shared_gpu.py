import os
import gc
import sys
import time
import shutil
import tempfile
import unittest
import subprocess
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.lib import shared_gpu
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.convolution_kernel import ConvolutionKernel


@unittest.skipIf(specula.cp is None, 'CUDA IPC needs a GPU')
class TestSharedGpu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.socket = os.path.join(cls.tmpdir, 'holder.sock')
        cls.holder = subprocess.Popen([sys.executable, '-m', 'specula.lib.shared_gpu',
                                       'serve', '--socket', cls.socket])
        for _ in range(300):
            if os.path.exists(cls.socket):
                break
            time.sleep(0.1)
        else:
            cls.holder.kill()
            raise RuntimeError('Holder did not start')

    @classmethod
    def tearDownClass(cls):
        shared_gpu._client_command(cls.socket, ('stop', True))
        cls.holder.wait(timeout=30)
        shutil.rmtree(cls.tmpdir)

    def setUp(self):
        self.env = patch.dict(os.environ, {shared_gpu.ENV_VAR: self.socket})
        self.env.start()
        shared_gpu._conn = None
        shared_gpu._conn_failed = False

    def tearDown(self):
        gc.collect()
        self.env.stop()

    def _entries(self):
        return {row[1]: row for row in shared_gpu._client_command(self.socket, ('list',))}

    def test_intmat_recmat(self):
        rng = np.random.default_rng(1)
        im_data = rng.standard_normal((40, 10)).astype(np.float32)
        rec_data = rng.standard_normal((10, 40)).astype(np.float32)
        im_file = os.path.join(self.tmpdir, 'im.fits')
        rec_file = os.path.join(self.tmpdir, 'rec.fits')
        Intmat(im_data, target_device_idx=0).save(im_file)
        Recmat(rec_data, target_device_idx=0).save(rec_file, overwrite=True)

        im1 = Intmat.restore(im_file, target_device_idx=0)
        im2 = Intmat.restore(im_file, target_device_idx=0)
        rec = Recmat.restore(rec_file, target_device_idx=0)

        self.assertTrue(shared_gpu.is_shared(im1.intmat))
        self.assertTrue(shared_gpu.is_shared(rec.recmat))
        # Same device memory, opened only once in this process
        self.assertEqual(im1.intmat.data.ptr, im2.intmat.data.ptr)
        np.testing.assert_array_equal(cpuArray(im1.intmat), im_data.astype(im1.dtype))
        np.testing.assert_array_equal(cpuArray(rec.recmat), rec_data.astype(rec.dtype))

        # Another process sees the same data
        code = ('import os, specula; specula.init(0, precision=%d); '
                'from specula.data_objects.intmat import Intmat; '
                'from specula.lib import shared_gpu; '
                'im = Intmat.restore(%r, target_device_idx=0); '
                'assert shared_gpu.is_shared(im.intmat); '
                'print(float(im.intmat.sum()))' % (specula.global_precision, im_file))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             env=os.environ.copy(), check=True)
        self.assertAlmostEqual(float(out.stdout.split()[-1]), float(im1.intmat.sum()), places=3)

        # Views stay shared, writes make a private copy
        im1.reduce_size(2)
        self.assertTrue(shared_gpu.is_shared(im1.intmat))
        im1.modes[0] = 0
        self.assertFalse(shared_gpu.is_shared(im1.intmat))
        self.assertEqual(im1.intmat.shape, (40, 8))
        np.testing.assert_array_equal(cpuArray(im2.intmat), im_data.astype(im2.dtype))
        rec.set_value(rec.recmat * 2)
        self.assertFalse(shared_gpu.is_shared(rec.recmat))
        np.testing.assert_array_equal(cpuArray(Recmat.restore(rec_file, target_device_idx=0).recmat),
                                      rec_data.astype(rec.dtype))

        # Calibration matrices are kept after all simulations released them
        del im1, im2, rec
        gc.collect()
        entries = self._entries()
        self.assertEqual(entries[os.path.abspath(im_file)][7], 0)
        self.assertTrue(entries[os.path.abspath(im_file)][8])

    def test_kernel(self):
        def make_kernel():
            return ConvolutionKernel(dimx=4, dimy=4, pxscale=0.1, pupil_size_m=8.0,
                                     dimension=32, launcher_pos=[5, 5, 0], seeing=1.0,
                                     zfocus=90e3, oversampling=1, return_fft=True,
                                     data_dir=self.tmpdir, target_device_idx=0)
        zlayer = [85e3, 90e3, 95e3]
        zprofile = [0.25, 0.5, 0.25]

        # First kernel: computed locally and saved
        local = make_kernel()
        local.prepare_for_sh(sodium_altitude=zlayer, sodium_intensity=zprofile)
        self.assertFalse(shared_gpu.is_shared(local.kernels))

        # Second one: processed by the holder from the saved file
        k = make_kernel()
        old = k.kernels
        k.prepare_for_sh(sodium_altitude=zlayer, sodium_intensity=zprofile)
        self.assertIsNot(k.kernels, old)
        self.assertTrue(shared_gpu.is_shared(k.kernels))
        np.testing.assert_allclose(cpuArray(k.kernels), cpuArray(local.kernels),
                                   rtol=1e-5, atol=1e-7)
        filename = os.path.abspath(os.path.join(self.tmpdir, k._kernel_fn + '.fits'))
        self.assertEqual(self._entries()[filename][7], 1)

        # A new sodium profile computes new private kernels, without
        # overwriting the shared ones
        shared_ptr = k.kernels.data.ptr
        k.prepare_for_sh(sodium_altitude=zlayer, sodium_intensity=[0.5, 0.25, 0.25])
        self.assertFalse(shared_gpu.is_shared(k.kernels))
        self.assertNotEqual(k.kernels.data.ptr, shared_ptr)

        # Kernels are freed by the holder when nobody uses them anymore
        gc.collect()
        self.assertNotIn(filename, self._entries())


if __name__ == '__main__':
    unittest.main()
