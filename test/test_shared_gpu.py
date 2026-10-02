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


def start_holder(directory, *args, check_interval=30):
    '''A holder of its own, so that the tests do not use a running one'''
    holder = subprocess.Popen([sys.executable, '-c',
                               'from specula.lib import shared_gpu; '
                               f'shared_gpu.DIR = {directory!r}; '
                               f'shared_gpu.CHECK_INTERVAL = {check_interval}; '
                               f'shared_gpu.main(["serve", *{list(args)!r}])'])
    with patch.object(shared_gpu, 'DIR', directory):
        for _ in range(300):
            if shared_gpu.holder_pid() == holder.pid:
                return holder
            time.sleep(0.1)
    holder.kill()
    raise RuntimeError('Holder did not start')


def stop_holder(directory, holder):
    with patch.object(shared_gpu, 'DIR', directory):
        shared_gpu.main(['stop'])
    holder.wait(timeout=30)


@unittest.skipIf(specula.cp is None, 'CUDA IPC needs a GPU')
class TestSharedGpu(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.dir = os.path.join(cls.tmpdir, 'holder')
        cls.holder = start_holder(cls.dir)

    @classmethod
    def tearDownClass(cls):
        stop_holder(cls.dir, cls.holder)
        shutil.rmtree(cls.tmpdir)

    def setUp(self):
        self.dir_patch = patch.object(shared_gpu, 'DIR', self.dir)
        self.dir_patch.start()

    def tearDown(self):
        gc.collect()
        self.dir_patch.stop()

    def _entries(self):
        return {info['file']: info for info in shared_gpu.list_arrays()}

    def test_holder_not_running(self):
        im_file = os.path.join(self.tmpdir, 'im_local.fits')
        Intmat(np.ones((4, 2), dtype=np.float32), target_device_idx=0).save(im_file)
        with patch.object(shared_gpu, 'DIR', os.path.join(self.tmpdir, 'none')):
            self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))

    def test_holder_stopped(self):
        directory = os.path.join(self.tmpdir, 'holder2')
        holder = start_holder(directory)
        im_file = os.path.join(self.tmpdir, 'im_stopped.fits')
        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        Intmat(data, target_device_idx=0).save(im_file)
        with patch.object(shared_gpu, 'DIR', directory):
            im = Intmat.restore(im_file, target_device_idx=0)
            self.assertTrue(shared_gpu.is_shared(im.intmat))
            published = {name: open(os.path.join(directory, name)).read()
                         for name in os.listdir(directory) if name.endswith('.json')}
            stop_holder(directory, holder)

            # The holder files are removed, the array in use is still valid
            self.assertEqual(os.listdir(directory), [])
            np.testing.assert_array_equal(cpuArray(im.intmat), data.astype(im.dtype))
            del im
            gc.collect()

            # New loads are local
            self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))

            # Also with stale files: a live PID and the handle of a freed array
            with open(os.path.join(directory, 'holder.pid'), 'w') as f:
                f.write(str(os.getpid()))
            for name, content in published.items():
                with open(os.path.join(directory, name), 'w') as f:
                    f.write(content)
            self.assertFalse(shared_gpu.is_shared(Intmat.restore(im_file, target_device_idx=0).intmat))

    def test_idle_timeout(self):
        directory = os.path.join(self.tmpdir, 'holder_idle')
        holder = start_holder(directory, '--idle-timeout', '0.02', check_interval=0.2)   # 1.2 s
        im_file = os.path.join(self.tmpdir, 'im_idle.fits')
        Intmat(np.ones((4, 2), dtype=np.float32), target_device_idx=0).save(im_file)
        try:
            with patch.object(shared_gpu, 'DIR', directory):
                def published():
                    return [info['users'] for info in shared_gpu.list_arrays()
                            if info['file'] == os.path.abspath(im_file)]

                # In use: kept beyond the idle time
                im = Intmat.restore(im_file, target_device_idx=0)
                self.assertTrue(shared_gpu.is_shared(im.intmat))
                time.sleep(2)
                self.assertEqual(published(), [[os.getpid()]])

                # Unused: freed after the idle time
                del im
                gc.collect()
                self.assertEqual(published(), [[]])
                time.sleep(2)
                self.assertEqual(published(), [])

                # A simulation that terminates is not a user anymore, even
                # if it does not remove its file
                code = ('import os, specula; specula.init(0, precision=%d); '
                        'from specula.data_objects.intmat import Intmat; '
                        'from specula.lib import shared_gpu; '
                        'shared_gpu.DIR = %r; '
                        'im = Intmat.restore(%r, target_device_idx=0); '
                        'assert shared_gpu.is_shared(im.intmat); '
                        'os._exit(0)' % (specula.global_precision, directory, im_file))
                subprocess.run([sys.executable, '-c', code], check=True)
                self.assertEqual(len(published()), 1)
                time.sleep(2)
                self.assertEqual(published(), [])
                self.assertEqual([n for n in os.listdir(directory) if n.endswith('.user')], [])
        finally:
            stop_holder(directory, holder)

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
                'shared_gpu.DIR = %r; '
                'im = Intmat.restore(%r, target_device_idx=0); '
                'assert shared_gpu.is_shared(im.intmat); '
                'print(float(im.intmat.sum()))' % (specula.global_precision, self.dir, im_file))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             check=True)
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

        # Arrays are kept after all simulations released them
        del im1, im2, rec
        gc.collect()
        self.assertIn(os.path.abspath(im_file), self._entries())

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

    def test_rewritten_file(self):
        im_file = os.path.join(self.tmpdir, 'im_rewritten.fits')
        Intmat(np.ones((4, 2), dtype=np.float32), target_device_idx=0).save(im_file)
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertEqual(float(im.intmat.sum()), 8)

        # The new version replaces the old one, which is still valid
        # for the simulations that use it
        time.sleep(0.01)
        Intmat(np.full((4, 2), 2, dtype=np.float32), target_device_idx=0).save(im_file)
        im2 = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im2.intmat))
        self.assertEqual(float(im2.intmat.sum()), 16)
        self.assertEqual(float(im.intmat.sum()), 8)
        rows = [info for info in shared_gpu.list_arrays() if info['file'] == os.path.abspath(im_file)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['mtime_ns'], os.stat(im_file).st_mtime_ns)

    def test_inotify(self):
        # The holder is woken up by inotify, not by polling
        watcher = shared_gpu._DirWatcher(self.tmpdir)
        self.assertIsNotNone(watcher.fd)
        t0 = time.monotonic()
        threading.Timer(0.3, lambda: open(os.path.join(self.tmpdir, 'wake'), 'w').close()).start()
        watcher.wait(10)
        self.assertTrue(0.25 < time.monotonic() - t0 < 5)
        os.close(watcher.fd)


if __name__ == '__main__':
    unittest.main()
