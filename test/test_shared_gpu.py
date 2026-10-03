import os
import gc
import sys
import json
import stat
import time
import shutil
import tempfile
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

# Simulation processes import specula from here: other tests can change the working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(specula.__file__)))


def sim_code(directory, body):
    '''Python code of a simulation process using *directory*'''
    return ('import os, sys, specula; specula.init(0, precision=%d)\n'
            'from specula.data_objects.intmat import Intmat\n'
            'from specula.lib import shared_gpu\n'
            'shared_gpu.DIR = %r\n' % (specula.global_precision, directory)) + body


@unittest.skipIf(specula.cp is None, 'CUDA IPC needs a GPU')
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

    def _published(self, filename):
        return [info for info in shared_gpu.list_arrays()
                if info['file'] == os.path.abspath(filename)]

    def _save_intmat(self, name, data):
        filename = os.path.join(self.tmpdir, name)
        Intmat(data, target_device_idx=0).save(filename, overwrite=True)
        return filename

    def test_intmat_recmat(self):
        rng = np.random.default_rng(1)
        im_data = rng.standard_normal((40, 10)).astype(np.float32)
        rec_data = rng.standard_normal((10, 40)).astype(np.float32)
        im_file = self._save_intmat('im.fits', im_data)
        rec_file = os.path.join(self.tmpdir, 'rec.fits')
        Recmat(rec_data, target_device_idx=0).save(rec_file, overwrite=True)

        im1 = Intmat.restore(im_file, target_device_idx=0)
        im2 = Intmat.restore(im_file, target_device_idx=0)
        rec = Recmat.restore(rec_file, target_device_idx=0)

        self.assertTrue(shared_gpu.is_shared(im1.intmat))
        self.assertTrue(shared_gpu.is_shared(rec.recmat))
        # Same device memory, loaded only once in this process
        self.assertEqual(im1.intmat.data.ptr, im2.intmat.data.ptr)
        np.testing.assert_array_equal(cpuArray(im1.intmat), im_data.astype(im1.dtype))
        np.testing.assert_array_equal(cpuArray(rec.recmat), rec_data.astype(rec.dtype))
        self.assertEqual(self._published(im_file)[0]['pid'], os.getpid())

        # Another process maps the same memory
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0)\n'
                                  'assert shared_gpu.is_shared(im.intmat)\n'
                                  'print(float(im.intmat.sum()))' % im_file)
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             check=True, cwd=ROOT)
        self.assertAlmostEqual(float(out.stdout.split()[-1]), float(im1.intmat.sum()), places=3)
        self.assertEqual(self._published(im_file)[0]['pid'], os.getpid())

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

        # Freed: loaded again on the next restore
        del im1, im2, rec
        gc.collect()
        published = self._published(im_file)[0]
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im.intmat))
        self.assertNotEqual(self._published(im_file)[0]['token'], published['token'])

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

    def test_publisher_frees(self):
        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        im_file = self._save_intmat('im_freed.fits', data)
        im = Intmat.restore(im_file, target_device_idx=0)

        # Another process maps the array, and reads it again after this
        # process has freed it
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0)\n'
                                  'assert shared_gpu.is_shared(im.intmat)\n'
                                  'print("mapped", flush=True)\n'
                                  'sys.stdin.readline()\n'
                                  'print(float(im.intmat.sum()))' % im_file)
        sim = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, cwd=ROOT)
        self.assertEqual(sim.stdout.readline().strip(), 'mapped')
        del im
        gc.collect()
        out, _ = sim.communicate('\n', timeout=60)
        self.assertEqual(float(out), data.sum())

    def test_stale_files(self):
        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        im_file = self._save_intmat('im_stale.fits', data)

        # A process that publishes the array and terminates without unpublishing it
        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0)\n'
                                  'os._exit(0)' % im_file)
        subprocess.run([sys.executable, '-c', code], check=True, cwd=ROOT)
        stale = self._published(im_file)
        self.assertEqual(len(stale), 1)
        self.assertNotEqual(stale[0]['pid'], os.getpid())

        # Its handle cannot be opened: loaded again, and published by this process
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im.intmat))
        np.testing.assert_array_equal(cpuArray(im.intmat), data.astype(im.dtype))
        self.assertEqual(self._published(im_file)[0]['pid'], os.getpid())
        [name] = os.listdir(self.dir)
        del im
        gc.collect()

        # Corrupted files
        for content in ['', '{"handle": "zz"}', json.dumps(dict(stale[0], handle='00' * 64))]:
            with open(os.path.join(self.dir, name), 'w') as f:
                f.write(content)
            im = Intmat.restore(im_file, target_device_idx=0)
            np.testing.assert_array_equal(cpuArray(im.intmat), data.astype(im.dtype))
            self.assertEqual(self._published(im_file)[0]['pid'], os.getpid())
            del im
            gc.collect()

    def test_reused_address(self):
        # This process publishes a file, frees it, and publishes another
        # one at the same address: the handle of the first file opens the
        # second one, the token tells that it is stale
        im_file1 = self._save_intmat('im_reused1.fits', np.full((400, 200), 1, dtype=np.float32))
        im_file2 = self._save_intmat('im_reused2.fits', np.full((400, 200), 2, dtype=np.float32))
        im1 = Intmat.restore(im_file1, target_device_idx=0)
        ptr = im1.intmat.data.ptr
        del im1
        gc.collect()
        im2 = Intmat.restore(im_file2, target_device_idx=0)
        self.assertEqual(im2.intmat.data.ptr, ptr)

        code = sim_code(self.dir, 'im = Intmat.restore(%r, target_device_idx=0)\n'
                                  'print(float(im.intmat.sum()))' % im_file1)
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                             check=True, cwd=ROOT)
        self.assertEqual(float(out.stdout.split()[-1]), 400 * 200)
        self.assertNotEqual(self._published(im_file1)[0]['pid'], os.getpid())
        self.assertEqual(float(im2.intmat.sum()), 2 * 400 * 200)

    def test_rewritten_file(self):
        im_file = self._save_intmat('im_rewritten.fits', np.ones((4, 2), dtype=np.float32))
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertEqual(float(im.intmat.sum()), 8)

        # The new version replaces the old one, which is still valid
        time.sleep(0.01)
        self._save_intmat('im_rewritten.fits', np.full((4, 2), 2, dtype=np.float32))
        im2 = Intmat.restore(im_file, target_device_idx=0)
        self.assertTrue(shared_gpu.is_shared(im2.intmat))
        self.assertEqual(float(im2.intmat.sum()), 16)
        self.assertEqual(float(im.intmat.sum()), 8)
        rows = self._published(im_file)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['mtime_ns'], os.stat(im_file).st_mtime_ns)

    def test_permissions(self):
        im_file = self._save_intmat('im_perm.fits', np.ones((4, 2), dtype=np.float32))
        im = Intmat.restore(im_file, target_device_idx=0)
        self.assertEqual(stat.S_IMODE(os.stat(self.dir).st_mode), 0o777)
        [name] = os.listdir(self.dir)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.dir, name)).st_mode), 0o666)
        del im

    def test_cannot_publish(self):
        # A directory that cannot be written: the array is loaded locally
        im_file = self._save_intmat('im_nodir.fits', np.ones((4, 2), dtype=np.float32))
        with patch.object(shared_gpu, 'DIR', os.path.join(im_file, 'not_a_dir')):
            im = Intmat.restore(im_file, target_device_idx=0)
        self.assertFalse(shared_gpu.is_shared(im.intmat))
        self.assertEqual(float(im.intmat.sum()), 8)


if __name__ == '__main__':
    unittest.main()
