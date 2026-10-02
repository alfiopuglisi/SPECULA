import os
import gc
import sys
import time
import shutil
import tempfile
import threading
import unittest
import subprocess
from unittest.mock import patch

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.lib import shared_gpu
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.m2c import M2C


def sim_code(directory, body):
    '''Python code of a simulation process using the sharing directory *directory*'''
    return ('import os, sys, time, specula; specula.init(0, precision=%d); '
            'from specula.data_objects.intmat import Intmat; '
            'from specula.lib import shared_gpu; '
            'shared_gpu.DIR = %r; ' % (specula.global_precision, directory) + body)


def wait_for(condition, timeout=30):
    t0 = time.monotonic()
    while not condition():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.05)


@unittest.skipIf(specula.cp is None or not shared_gpu._vmm_supported(0),
                 'needs a GPU with VMM and POSIX file descriptors')
class TestSharedGpu(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dir = os.path.join(self.tmpdir, 'shared')
        self.dir_patch = patch.object(shared_gpu, 'DIR', self.dir)
        self.dir_patch.start()

    def tearDown(self):
        gc.collect()
        self.dir_patch.stop()
        shutil.rmtree(self.tmpdir)

    def _intmat_file(self, name, data=None):
        filename = os.path.join(self.tmpdir, name)
        if data is None:
            data = np.arange(8, dtype=np.float32).reshape(4, 2)
        Intmat(data, target_device_idx=0).save(filename)
        return filename

    def test_intmat_recmat(self):
        rng = np.random.default_rng(1)
        im_data = rng.standard_normal((40, 10)).astype(np.float32)
        rec_data = rng.standard_normal((10, 40)).astype(np.float32)
        im_file = self._intmat_file('im.fits', im_data)
        rec_file = os.path.join(self.tmpdir, 'rec.fits')
        Recmat(rec_data, target_device_idx=0).save(rec_file, overwrite=True)

        im1 = Intmat.restore(im_file, target_device_idx=0)
        im2 = Intmat.restore(im_file, target_device_idx=0)
        rec = Recmat.restore(rec_file, target_device_idx=0)

        self.assertTrue(shared_gpu.is_shared(im1.intmat))
        self.assertTrue(shared_gpu.is_shared(rec.recmat))
        # Same device memory, mapped only once in this process
        self.assertEqual(im1.intmat.data.ptr, im2.intmat.data.ptr)
        np.testing.assert_array_equal(cpuArray(im1.intmat), im_data.astype(im1.dtype))
        np.testing.assert_array_equal(cpuArray(rec.recmat), rec_data.astype(rec.dtype))

        # Another process maps the same data, without loading it
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0); '
                                  'assert shared_gpu.is_shared(im.intmat); '
                                  'print(float(im.intmat.sum()))' % im_file)
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        self.assertAlmostEqual(float(out.stdout.split()[-1]), float(im1.intmat.sum()), places=3)
        self.assertEqual(shared_gpu.list_arrays()[0]['loaded_by'], os.getpid())

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

    def test_owner_change(self):
        # A loads, this process maps it, A terminates, B maps it from this process
        im_file = self._intmat_file('im_owner.fits')
        release = os.path.join(self.tmpdir, 'release')
        a = subprocess.Popen([sys.executable, '-c', sim_code(
            self.dir, 'im = Intmat.restore(%r, target_device_idx=0); '
                      'assert shared_gpu.is_shared(im.intmat); '
                      'exec("while not os.path.exists(%r): time.sleep(0.05)")' % (im_file, release))])
        wait_for(lambda: shared_gpu.list_arrays())
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im.intmat))
        self.assertEqual(shared_gpu.list_arrays()[0]['loaded_by'], a.pid)
        open(release, 'w').close()
        a.wait(timeout=30)

        self.assertEqual(shared_gpu.list_arrays()[0]['holders'], [os.getpid()])
        np.testing.assert_array_equal(cpuArray(im.intmat), np.arange(8).reshape(4, 2))
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0); '
                                  'assert shared_gpu.is_shared(im.intmat); '
                                  'assert shared_gpu.list_arrays()[0]["loaded_by"] == %d; '
                                  'print(float(im.intmat.sum()))' % (im_file, a.pid))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        self.assertEqual(float(out.stdout.split()[-1]), 28)

    def test_without_keeper(self):
        # When all the holders have terminated, the array is loaded again
        im_file = self._intmat_file('im_nokeeper.fits')
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0); '
                                  'assert shared_gpu.is_shared(im.intmat)' % im_file)
        subprocess.run([sys.executable, '-c', code], check=True)
        self.assertEqual(shared_gpu.list_arrays(), [])
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im.intmat))
        self.assertEqual(shared_gpu.list_arrays()[0]['loaded_by'], os.getpid())

    def test_concurrent_start(self):
        # Simulations started together load the array only once
        im_file = self._intmat_file('im_concurrent.fits')
        start = os.path.join(self.tmpdir, 'start')
        release = os.path.join(self.tmpdir, 'release')
        code = sim_code(self.dir, 'exec("while not os.path.exists(%r): time.sleep(0.01)"); '
                                  'im = Intmat.restore(%r, target_device_idx=0); '
                                  'assert shared_gpu.is_shared(im.intmat); '
                                  'print(shared_gpu.list_arrays()[0]["id"], flush=True); '
                                  'exec("while not os.path.exists(%r): time.sleep(0.05)")'
                                  % (start, im_file, release))
        sims = [subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
                for _ in range(3)]
        time.sleep(3)    # imports and CUDA initialization
        open(start, 'w').close()
        ids = [sim.stdout.readline().strip() for sim in sims]
        open(release, 'w').close()
        for sim in sims:
            sim.wait(timeout=30)
            self.assertEqual(sim.returncode, 0)
        self.assertEqual(len(set(ids)), 1)

    def test_keeper(self):
        im_file = self._intmat_file('im_keeper.fits')
        keeper = subprocess.Popen([sys.executable, '-c',
                                   'from specula.lib import shared_gpu; '
                                   f'shared_gpu.DIR = {self.dir!r}; '
                                   'shared_gpu.CHECK_INTERVAL = 0.2; '
                                   'shared_gpu.main(["keep", "--idle-timeout", "0.02"])'])   # 1.2 s
        try:
            wait_for(lambda: shared_gpu.keeper_pid() == keeper.pid)
            # The keeper does not use CUDA
            with open(f'/proc/{keeper.pid}/maps') as f:
                self.assertNotIn('libcuda', f.read())

            # The array of a terminated simulation is kept
            code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0); '
                                      'assert shared_gpu.is_shared(im.intmat); '
                                      'exec("while not shared_gpu.list_arrays()[0][\'holders\'] == '
                                      'sorted([os.getpid(), %d]): time.sleep(0.05)")' % (im_file, keeper.pid))
            sim = subprocess.run([sys.executable, '-c', code], timeout=60)
            self.assertEqual(sim.returncode, 0)
            self.assertEqual(shared_gpu.list_arrays()[0]['holders'], [keeper.pid])

            # and mapped from the keeper, without loading it again
            im = Intmat.restore(im_file, target_device_idx=0)
            self.assertTrue(shared_gpu.is_shared(im.intmat))
            info = shared_gpu.list_arrays()[0]
            self.assertNotEqual(info['loaded_by'], os.getpid())
            np.testing.assert_array_equal(cpuArray(im.intmat), np.arange(8).reshape(4, 2))

            # In use: kept beyond the idle time
            time.sleep(2)
            self.assertEqual(shared_gpu.list_arrays()[0]['holders'], sorted([os.getpid(), keeper.pid]))

            # Unused: released after the idle time
            del im
            gc.collect()
            wait_for(lambda: shared_gpu.list_arrays() == [], timeout=10)
            self.assertFalse([n for n in os.listdir(self.dir) if n.endswith('.json')])
        finally:
            shared_gpu.main(['stop'])
            keeper.wait(timeout=30)
        self.assertIsNone(shared_gpu.keeper_pid())

    def test_rewritten_file(self):
        im_file = self._intmat_file('im_rewritten.fits', np.ones((4, 2), dtype=np.float32))
        im = Intmat.restore(im_file, target_device_idx=0)
        time.sleep(0.01)
        self._intmat_file('im_rewritten.fits', np.full((4, 2), 2, dtype=np.float32))
        im2 = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im2.intmat))
        self.assertEqual(float(im2.intmat.sum()), 16)
        self.assertEqual(float(im.intmat.sum()), 8)

    def test_not_supported(self):
        im_file = self._intmat_file('im_local.fits')
        with patch.dict(shared_gpu._supported, {0: False}):
            self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))

    def test_inotify(self):
        watcher = shared_gpu._DirWatcher(self.tmpdir)
        self.assertIsNotNone(watcher.fd)
        t0 = time.monotonic()
        threading.Timer(0.3, lambda: open(os.path.join(self.tmpdir, 'wake'), 'w').close()).start()
        watcher.wait(10)
        self.assertTrue(0.25 < time.monotonic() - t0 < 5)
        os.close(watcher.fd)


if __name__ == '__main__':
    unittest.main()
