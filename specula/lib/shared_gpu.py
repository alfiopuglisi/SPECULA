'''
Sharing of read-only GPU arrays between simulation processes (CUDA IPC).

Several simulations running on the same GPU usually load the same large
arrays (interaction and reconstruction matrices, LGS convolution kernels).
With this module, a single *holder* process loads each array once and
exports it with a CUDA IPC handle; the simulations map the same device
memory instead of allocating and loading their own copy.

Usage::

    # start the holder (once per machine, it serves all GPUs)
    python -m specula.lib.shared_gpu serve &

    # enable sharing in the simulations
    export SPECULA_SHARED_GPU=1        # default socket, or a socket path
    specula params.yml ...

    python -m specula.lib.shared_gpu list    # show the loaded arrays
    python -m specula.lib.shared_gpu stop    # stop the holder

When SPECULA_SHARED_GPU is not set, or the holder is not running, or the
object is on the CPU, everything is loaded locally as usual.

Notes
-----
- Shared arrays are read-only by convention: CuPy cannot enforce it, and
  an in-place write is seen by all the simulations. The data objects
  that use shared arrays make a private copy before writing into them
  (copy-on-write), but views of them must not be modified in place
  (for example ``intmat.modes[3:5] += 1``).
- The holder must stay alive while simulations use its arrays.
  Calibration matrices (kind 'fits') are kept until the holder is
  stopped, so that later simulations find them already loaded. Kernels
  are freed when no simulation uses them anymore, since time-varying
  sodium profiles generate many of them.
- Arrays are identified by file path, modification time, size and
  precision: a file that is rewritten is loaded again.
- GPUs are identified by PCI bus id, so that the holder and the
  simulations can have different CUDA_VISIBLE_DEVICES.
'''

import os
import sys
import stat
import argparse
import tempfile
import threading
import weakref
from contextlib import contextmanager
from multiprocessing.connection import Listener, Client

import numpy as np

import specula

ENV_VAR = 'SPECULA_SHARED_GPU'

KIND_FITS = 'fits'
KIND_KERNEL_FFT = 'kernel_fft'


def default_socket_path():
    base = os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir()
    return os.path.join(base, f'specula_shared_gpu_{os.getuid()}.sock')


def socket_path():
    '''Socket of the holder, or None if sharing is disabled'''
    value = os.environ.get(ENV_VAR, '').strip()
    if value.lower() in ('', '0', 'false', 'no', 'off'):
        return None
    if value.lower() in ('1', 'true', 'yes', 'on', 'default'):
        return default_socket_path()
    return value


def _file_id(filename):
    st = os.stat(filename)
    return os.path.abspath(filename), st.st_mtime_ns, st.st_size


# ---------------------------------------------------------------------------
# Client side (simulation processes)

# Reentrant, since _IpcMapping.__del__() can be called by the garbage collector
_lock = threading.RLock()
_conn = None
_conn_failed = False
_mappings = weakref.WeakValueDictionary()   # key -> _IpcMapping
_mappings_by_ptr = weakref.WeakValueDictionary()   # (device_id, ptr) -> _IpcMapping


class _IpcMapping:
    '''
    A device memory region opened from an IPC handle.
    It is the owner of the UnownedMemory of the arrays built on it,
    so it is closed (and released in the holder) when the last
    array or view referencing it is garbage collected.
    '''
    def __init__(self, key, handle, shape, dtype, nbytes, device_id):
        self.key = key
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = nbytes
        self.device_id = device_id
        cp = specula.cp
        with cp.cuda.Device(device_id):
            self.ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)

    def __del__(self):
        try:
            if specula.cp.cuda.get_current_stream().is_capturing():
                # The garbage collector can run during a CUDA graph capture,
                # where synchronizing is not allowed: close it later
                _pending_close.append((self.key, self.ptr, self.device_id))
            else:
                _close(self.key, self.ptr, self.device_id)
        except Exception:
            # Interpreter shutdown: the driver and the holder clean up anyway
            pass


_pending_close = []


def _close(key, ptr, device_id):
    cp = specula.cp
    with cp.cuda.Device(device_id):
        # Work queued on the memory must be completed before unmapping it
        cp.cuda.runtime.deviceSynchronize()
        cp.cuda.runtime.ipcCloseMemHandle(ptr)
    _request(('release', key))


def _connect():
    global _conn, _conn_failed
    if _conn is None and not _conn_failed:
        path = socket_path()
        try:
            _conn = Client(path, family='AF_UNIX')
        except OSError as e:
            _conn_failed = True
            specula.get_specula_logger(__name__).warning(
                f'{ENV_VAR} is set, but the shared GPU array holder is not reachable '
                f'at {path} ({e}): arrays will be loaded locally')
    return _conn


def _request(msg):
    with _lock:
        conn = _connect()
        if conn is None:
            return None
        conn.send(msg)
        status, payload = conn.recv()
    if status != 'ok':
        raise RuntimeError(f'Shared GPU array holder: {payload}')
    return payload


def get_shared_array(kind, filename, target_device_idx, precision, **params):
    '''
    Get a shared, read-only array from the holder.

    Parameters
    ----------
    kind: str
        KIND_FITS: image extension *exten* of a FITS file, as returned by
        specula.lib.fits_io.load_fits_array().
        KIND_KERNEL_FFT: processed kernels (``ConvolutionKernel.kernels``
        with return_fft=True) of a ConvolutionKernel FITS file.
    filename: str
        FITS file name
    target_device_idx: int
        device of the array (None for the default device)
    precision: int
        SPECULA precision (None for the global precision)
    params:
        additional parameters of *kind* (exten for KIND_FITS)

    Returns
    -------
    cupy.ndarray, or None if sharing is disabled, the holder is not
    reachable or the array is on the CPU. In this case the caller
    must load the array itself.
    '''
    if socket_path() is None:
        return None
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx
    if target_device_idx < 0 or specula.cp is None:
        return None
    if precision is None:
        precision = specula.global_precision

    cp = specula.cp
    pci_bus_id = cp.cuda.Device(target_device_idx).pci_bus_id
    params = tuple(sorted(params.items()))
    key = (kind, _file_id(filename), precision, pci_bus_id, params)

    with _lock:
        while _pending_close:
            _close(*_pending_close.pop())
        mapping = _mappings.get(key)
        if mapping is None:
            # One holder reference per process, released when the mapping is closed
            payload = _request(('get', key))
            if payload is None:
                return None
            mapping = _IpcMapping(key, *payload, device_id=target_device_idx)
            _mappings[key] = mapping
            _mappings_by_ptr[(target_device_idx, mapping.ptr)] = mapping

    mem = cp.cuda.UnownedMemory(mapping.ptr, mapping.nbytes, owner=mapping,
                                device_id=target_device_idx)
    return cp.ndarray(mapping.shape, dtype=mapping.dtype,
                      memptr=cp.cuda.MemoryPointer(mem, 0))


def is_shared(arr):
    '''True if *arr* (or the array it is a view of) is a shared array'''
    cp = specula.cp
    if cp is None or not isinstance(arr, cp.ndarray):
        return False
    mem = arr.data.mem
    # UnownedMemory does not expose its owner: the mapping is found by
    # pointer, it is alive as long as the memory references it
    return isinstance(mem, cp.cuda.UnownedMemory) and \
        (mem.device_id, mem.ptr) in _mappings_by_ptr


def writable(arr):
    '''
    Return *arr*, or a private copy of it if it is a shared array.
    To be used by the data objects before writing into their arrays.
    '''
    return arr.copy() if is_shared(arr) else arr


# ---------------------------------------------------------------------------
# Holder side

class _Entry:
    def __init__(self, arr, keep):
        cp = specula.cp
        self.arr = arr
        self.handle = cp.cuda.runtime.ipcGetMemHandle(arr.data.ptr)
        self.keep = keep
        self.refs = 0


@contextmanager
def _dedicated_allocation(nbytes):
    '''
    Inside this context, the first allocation of *nbytes* bytes on the
    current device is a dedicated cudaMalloc() instead of a sub-allocation
    of the CuPy memory pool, so that it can be exported with a IPC handle
    of its own. All the other allocations use the memory pool.
    '''
    cp = specula.cp
    pool = cp.get_default_memory_pool()
    done = []

    def alloc(size):
        if size == nbytes and not done:
            done.append(True)
            return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)
        return pool.malloc(size)

    with cp.cuda.using_allocator(alloc):
        yield


def _exportable(arr, nbytes):
    '''Make sure that *arr* owns a whole dedicated cudaMalloc() allocation'''
    cp = specula.cp
    mem = arr.data.mem
    if type(mem) is cp.cuda.Memory and arr.data.ptr == mem.ptr and \
            arr.flags.c_contiguous and arr.nbytes == nbytes:
        return arr
    # Fallback: copy into a dedicated allocation
    out = cp.ndarray(arr.shape, arr.dtype, cp.cuda.MemoryPointer(cp.cuda.Memory(arr.nbytes), 0))
    out[...] = arr
    return out


def _load_fits(filename, device_idx, precision, exten=1):
    from astropy.io import fits
    from specula.lib.fits_io import load_fits_array, _target_dtype, _BITPIX2DTYPE

    with fits.open(filename) as hdul:
        hdr = hdul[exten].header
        shape = hdul[exten].shape
        bitpix = hdr.get('BITPIX')
    # Expected dtype of the result. If it is different (for example for
    # scaled data), _exportable() makes a copy.
    if bitpix in _BITPIX2DTYPE and hdr.get('BSCALE', 1) == 1 and hdr.get('BZERO', 0) == 0:
        dtype = _target_dtype(np.dtype(_BITPIX2DTYPE[bitpix]), precision)
    else:
        dtype = np.dtype(specula.cpu_float_dtype_list[precision])
    nbytes = int(np.prod(shape)) * dtype.itemsize
    with _dedicated_allocation(nbytes):
        arr = load_fits_array(filename, exten, device_idx, precision)
    return _exportable(arr, arr.nbytes)


def _load_kernel_fft(filename, device_idx, precision):
    from astropy.io import fits
    from specula.lib.fits_io import load_fits_array
    from specula.data_objects.convolution_kernel import ConvolutionKernel

    hdr = fits.getheader(filename, ext=0)
    shape = (hdr['DIMX'] * hdr['DIMY'], hdr['DIM'], hdr['DIM'] // 2 + 1)
    complex_dtype = np.dtype(specula.cpu_complex_dtype_list[precision])
    nbytes = int(np.prod(shape)) * complex_dtype.itemsize
    # from_header() allocates the kernels, that process_kernels() fills in place
    with _dedicated_allocation(nbytes):
        kernel_obj = ConvolutionKernel.from_header(hdr, target_device_idx=device_idx,
                                                   precision=precision)
    kernel_obj.real_kernels = load_fits_array(filename, 1, device_idx, precision)
    kernel_obj.process_kernels(return_fft=True)
    return _exportable(kernel_obj.kernels, nbytes)


class Holder:
    '''Loads arrays on request and keeps them for the clients'''

    def __init__(self, path, logger):
        self.path = path
        self.logger = logger
        self.entries = {}
        self.lock = threading.Lock()
        self.key_locks = {}
        self.listener = None
        self.stopping = False

    def _get(self, key, conn_refs):
        kind, (filename, mtime_ns, size), precision, pci_bus_id, params = key
        params = dict(params)
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        # One load per key, other keys can be loaded concurrently
        with key_lock:
            with self.lock:
                entry = self.entries.get(key)
            if entry is None:
                if _file_id(filename) != (filename, mtime_ns, size):
                    raise ValueError(f'{filename} changed while loading it')
                cp = specula.cp
                device_idx = cp.cuda.runtime.deviceGetByPCIBusId(pci_bus_id)
                self.logger.info(f'Loading {kind} {filename} on GPU {pci_bus_id}')
                with cp.cuda.Device(device_idx):
                    if kind == KIND_FITS:
                        arr = _load_fits(filename, device_idx, precision, **params)
                        keep = True
                    elif kind == KIND_KERNEL_FFT:
                        arr = _load_kernel_fft(filename, device_idx, precision)
                        keep = False
                    else:
                        raise ValueError(f'Unknown kind {kind}')
                    cp.cuda.runtime.deviceSynchronize()
                    # Release the temporaries of the loading
                    cp.get_default_memory_pool().free_all_blocks()
                    entry = _Entry(arr, keep)
                self.logger.info(f'Loaded {filename}: {arr.shape} {arr.dtype}, '
                                 f'{arr.nbytes / 2**20:.1f} MiB')
            with self.lock:
                self.entries[key] = entry
                entry.refs += 1
                conn_refs[key] = conn_refs.get(key, 0) + 1
        arr = entry.arr
        return entry.handle, arr.shape, arr.dtype.str, arr.nbytes

    def _release(self, key, conn_refs, count=1):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return
            entry.refs -= count
            conn_refs[key] = conn_refs.get(key, 0) - count
            if conn_refs[key] <= 0:
                del conn_refs[key]
            if entry.refs <= 0 and not entry.keep:
                self.logger.info(f'Freeing {key[1][0]}')
                del self.entries[key]

    def _list(self):
        with self.lock:
            return [(key[0], key[1][0], key[2], key[3], e.arr.shape, e.arr.dtype.str,
                     e.arr.nbytes, e.refs, e.keep) for key, e in self.entries.items()]

    def _serve_connection(self, conn):
        conn_refs = {}
        try:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    break
                try:
                    cmd = msg[0]
                    if cmd == 'get':
                        reply = self._get(msg[1], conn_refs)
                    elif cmd == 'release':
                        reply = self._release(msg[1], conn_refs)
                    elif cmd == 'list':
                        reply = self._list()
                    elif cmd == 'stop':
                        force = msg[1]
                        with self.lock:
                            in_use = sum(e.refs for e in self.entries.values())
                        if in_use and not force:
                            raise RuntimeError(f'{in_use} arrays are in use by simulations, '
                                               'use --force to stop anyway')
                        self.stopping = True
                        reply = None
                    else:
                        raise ValueError(f'Unknown command {cmd}')
                    conn.send(('ok', reply))
                except Exception as e:
                    self.logger.exception('Error serving request')
                    conn.send(('error', f'{type(e).__name__}: {e}'))
                if self.stopping:
                    # Unblock accept() in serve()
                    try:
                        Client(self.path, family='AF_UNIX').close()
                    except OSError:
                        pass
                    break
        finally:
            # A simulation that terminates releases all its arrays
            for key, count in list(conn_refs.items()):
                self._release(key, conn_refs, count)
            conn.close()

    def serve(self):
        if os.path.exists(self.path):
            try:
                Client(self.path, family='AF_UNIX').close()
                raise RuntimeError(f'A holder is already running on {self.path}')
            except ConnectionRefusedError:
                os.unlink(self.path)   # stale socket
        old_umask = os.umask(0o077)
        try:
            # Only the user can connect: the protocol uses pickle
            self.listener = Listener(self.path, family='AF_UNIX')
        finally:
            os.umask(old_umask)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        self.logger.info(f'Shared GPU array holder listening on {self.path}')
        try:
            while not self.stopping:
                conn = self.listener.accept()
                if self.stopping:
                    conn.close()
                    break
                threading.Thread(target=self._serve_connection, args=(conn,),
                                 daemon=True).start()
        finally:
            self.listener.close()
            self.logger.info('Shared GPU array holder stopped')


def _client_command(path, msg):
    with Client(path, family='AF_UNIX') as conn:
        conn.send(msg)
        status, payload = conn.recv()
    if status != 'ok':
        raise SystemExit(payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m specula.lib.shared_gpu',
                                     description='Holder of GPU arrays shared between '
                                                 'SPECULA simulations')
    parser.add_argument('command', choices=['serve', 'list', 'stop'])
    parser.add_argument('--socket', default=None,
                        help=f'socket path (default: ${ENV_VAR} if it is a path, '
                             f'otherwise {default_socket_path()})')
    parser.add_argument('--force', action='store_true',
                        help='stop even if simulations are using the arrays')
    args = parser.parse_args(argv)

    path = args.socket or socket_path() or default_socket_path()

    if args.command == 'serve':
        # The holder serves all GPUs, with explicit device and precision in each request
        specula.init(0, precision=1)
        Holder(path, specula.get_specula_logger(__name__)).serve()
    elif args.command == 'list':
        rows = _client_command(path, ('list',))
        total = 0
        for kind, filename, precision, bus, shape, dtype, nbytes, refs, keep in rows:
            total += nbytes
            print(f'{kind:10s} {bus} {nbytes / 2**20:10.1f} MiB  refs={refs} '
                  f'{"keep" if keep else "    "} {shape} {dtype} {filename}')
        print(f'{len(rows)} arrays, {total / 2**30:.2f} GiB')
    elif args.command == 'stop':
        _client_command(path, ('stop', args.force))


if __name__ == '__main__':
    sys.exit(main())
