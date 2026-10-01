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
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.m2c import M2C


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
        # Do not leave a connection to this holder to the other tests
        if shared_gpu._conn is not None:
            shared_gpu._conn.close()
        shared_gpu._conn = None
        shared_gpu._conn_failed = False

    def test_holder_terminated(self):
        # Without a reachable holder, arrays are loaded locally
        shared_gpu._conn = None
        with patch.dict(os.environ, {shared_gpu.ENV_VAR: os.path.join(self.tmpdir, 'none.sock')}):
            im_file = os.path.join(self.tmpdir, 'im_local.fits')
            Intmat(np.ones((4, 2), dtype=np.float32), target_device_idx=0).save(im_file)
            self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))
        # A connection that breaks is the same
        shared_gpu._conn_failed = False
        broken = shared_gpu.Client(self.socket, family='AF_UNIX')
        broken.close()
        shared_gpu._conn = broken
        self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))

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

    def test_ifunc_m2c(self):
        rng = np.random.default_rng(2)
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[1:7, 1:7] = 1
        npoints = int(mask.sum())
        if_data = rng.standard_normal((5, npoints)).astype(np.float32)
        inv_data = rng.standard_normal((npoints, 5)).astype(np.float32)
        m2c_data = rng.standard_normal((20, 5)).astype(np.float32)
        if_file = os.path.join(self.tmpdir, 'ifunc.fits')
        inv_file = os.path.join(self.tmpdir, 'ifunc_inv.fits')
        m2c_file = os.path.join(self.tmpdir, 'm2c.fits')
        IFunc(if_data, mask=mask, target_device_idx=0).save(if_file, overwrite=True)
        IFuncInv(inv_data, mask=mask, target_device_idx=0).save(inv_file, overwrite=True)
        M2C(m2c_data, target_device_idx=0).save(m2c_file, overwrite=True)

        ifunc = IFunc.restore(if_file, target_device_idx=0)
        ifunc_inv = IFuncInv.restore(inv_file, target_device_idx=0)
        m2c = M2C.restore(m2c_file, target_device_idx=0)
        for obj, arr, ref in [(ifunc, ifunc.influence_function, if_data),
                              (ifunc_inv, ifunc_inv.ifunc_inv, inv_data),
                              (m2c, m2c.m2c, m2c_data)]:
            self.assertTrue(shared_gpu.is_shared(arr), type(obj).__name__)
            np.testing.assert_array_equal(cpuArray(arr), ref.astype(obj.dtype))

        # Mode cuts are views and stay shared, writes make a private copy
        m2c.set_nmodes(3)
        self.assertTrue(shared_gpu.is_shared(m2c.m2c))
        for obj in [ifunc, ifunc_inv, m2c]:
            obj.set_value(obj.get_value() * 0)
            self.assertFalse(shared_gpu.is_shared(obj.get_value()), type(obj).__name__)
        np.testing.assert_array_equal(
            cpuArray(IFunc.restore(if_file, target_device_idx=0).influence_function),
            if_data.astype(ifunc.dtype))
        np.testing.assert_array_equal(
            cpuArray(M2C.restore(m2c_file, target_device_idx=0).m2c), m2c_data.astype(m2c.dtype))

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
