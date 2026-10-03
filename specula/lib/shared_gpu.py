'''
Sharing of read-only GPU arrays between simulation processes (CUDA IPC).

Several simulations running on the same GPU usually load the same large
calibration arrays from FITS files (interaction and reconstruction matrices,
influence functions, modes-to-commands matrices). With this module, the
first simulation that loads an array publishes a CUDA IPC handle of it,
and the others map the same device memory instead of allocating and
loading their own copy.

Sharing is done by specula.lib.fits_io.load_fits_array(), used by the
restore() methods of the data objects, and is transparent for its users.
There is no process to start: each array is published in a file
<key>.json of a directory shared by all users (DIR), with its IPC handle
and the modification time and size of the FITS file. <key> is a hash of
the file name, FITS extension, precision and GPU, so that a simulation
finds the array of a file from the file name alone. If the file cannot
be read, the file has changed, or the handle cannot be opened, the
simulation loads the array itself and publishes it again.

    python -m specula.lib.shared_gpu    # show the arrays published by running processes

Notes
-----
- Shared arrays are read-only by convention: CuPy cannot enforce it, and
  an in-place write is seen by all the simulations. The data objects
  that use shared arrays make a private copy before writing into them
  (copy-on-write), but views of them must not be modified in place
  (for example ``intmat.modes[3:5] += 1``).
- The CUDA driver keeps the device memory alive while any process maps
  it, even after the process that loaded it has freed it or terminated.
  But a handle cannot be opened anymore after that: the simulations
  started later load the array again. Files are never removed, since
  stale handles are detected (they are small, one per FITS file,
  extension, precision and GPU).
- Published arrays are allocated with their own cudaMalloc(), outside of
  the CuPy memory pool, so that their memory is never reused for other
  data while the handle is published.
- GPUs are identified by PCI bus id, since CUDA_VISIBLE_DEVICES can
  change device numbers between processes.
- DIR is writable by all users, without the sticky bit, so that any user
  can replace a stale file of another user. As a consequence, any user
  can make the simulations of another user map their arrays: use it only
  on machines whose users trust each other.
'''

import os
import sys
import json
import hashlib
import tempfile
import weakref

import numpy as np

import specula

# Directory of the published arrays, shared by all users. /dev/shm is a RAM filesystem
DIR = os.path.join('/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir(),
                   'specula_shared_gpu')


def _path(name):
    return os.path.join(DIR, name)


def _read_json(name):
    try:
        with open(_path(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _publish(name, info):
    '''Write a file of DIR atomically, writable by all users'''
    if not os.path.isdir(DIR):
        os.makedirs(DIR, exist_ok=True)
        # The umask does not apply to chmod. Only the owner can do it.
        try:
            os.chmod(DIR, 0o777)
        except PermissionError:
            pass
    tmp = _path(f'.{name}.{os.getpid()}.tmp')
    try:
        with open(tmp, 'w') as f:
            json.dump(info, f)
        os.chmod(tmp, 0o666)
        os.replace(tmp, _path(name))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# Bytes before the data of a shared array, starting with the random 8-byte
# token of its file. A handle can also open the memory allocated and published
# later at the same address: the token tells if it is the right one. 256 bytes
# keep the data aligned as allocated by cudaMalloc().
_HEADER = 256

_by_key = weakref.WeakValueDictionary()   # key -> _SharedMemory
_by_ptr = weakref.WeakValueDictionary()   # (device_id, ptr) -> _SharedMemory
_capture_leftovers = []                   # memory released during a CUDA graph capture


class _SharedMemory:
    '''
    Device memory of a shared array: either loaded by this process and
    published (*memory* is its cupy.cuda.Memory), or opened from the
    handle published by another process (*info*). It is the owner of the
    UnownedMemory of the arrays built on it, so it is freed or closed
    when the last array or view referencing it is garbage collected.
    '''
    def __init__(self, info, device_id, memory=None):
        cp = specula.cp
        self.info = info
        self.device_id = device_id
        self.memory = memory
        if memory is not None:
            self.base = memory.ptr
            return
        with cp.cuda.Device(device_id):
            self.base = cp.cuda.runtime.ipcOpenMemHandle(bytes.fromhex(info['handle']))
            token = np.empty(len(info['token']) // 2, dtype=np.uint8)
            cp.cuda.runtime.memcpy(token.ctypes.data, self.base, token.nbytes,
                                   cp.cuda.runtime.memcpyDeviceToHost)
            if token.tobytes().hex() != info['token']:
                cp.cuda.runtime.ipcCloseMemHandle(self.base)
                raise ValueError('stale handle')

    @property
    def ptr(self):
        return self.base + _HEADER

    def array(self):
        cp = specula.cp
        mem = cp.cuda.UnownedMemory(self.ptr, self.info['nbytes'], owner=self,
                                    device_id=self.device_id)
        return cp.ndarray(tuple(self.info['shape']), dtype=np.dtype(self.info['dtype']),
                          memptr=cp.cuda.MemoryPointer(mem, 0))

    def __del__(self):
        try:
            cp = specula.cp
            # The garbage collector can run during a CUDA graph capture, where
            # synchronizing and freeing are not allowed: the memory is then
            # kept until the process terminates
            if cp.cuda.get_current_stream().is_capturing():
                _capture_leftovers.append(self.memory)
            elif self.memory is None:
                with cp.cuda.Device(self.device_id):
                    # Work queued on the memory must be completed before unmapping it
                    cp.cuda.runtime.deviceSynchronize()
                    cp.cuda.runtime.ipcCloseMemHandle(self.base)
            # The memory loaded by this process is freed with self.memory.
            # Its file is not removed, its handle becomes stale.
        except Exception:
            # Interpreter shutdown: the driver cleans up anyway
            pass


def _load(request, device_id):
    '''
    Load the array with dedicated cudaMalloc() allocations instead of
    sub-allocations of the CuPy memory pool, so that it owns a whole
    allocation that can be exported with a IPC handle of its own,
    after a header for the token.
    '''
    from specula.lib.fits_io import load_fits_array
    cp = specula.cp

    def alloc(size):
        return cp.cuda.MemoryPointer(cp.cuda.Memory(size + _HEADER), _HEADER)

    with cp.cuda.Device(device_id), cp.cuda.using_allocator(alloc):
        arr = load_fits_array(request['file'], request['exten'], device_id,
                              request['precision'], shared=False)
        if arr.data.ptr != arr.data.mem.ptr + _HEADER or not arr.flags.c_contiguous:
            arr = arr.copy()
        cp.cuda.runtime.deviceSynchronize()
    return arr


def _publish_array(key, request, arr, device_id):
    cp = specula.cp
    token = np.frombuffer(os.urandom(8), dtype=np.uint8)
    with cp.cuda.Device(device_id):
        cp.cuda.runtime.memcpy(arr.data.mem.ptr, token.ctypes.data, token.nbytes,
                               cp.cuda.runtime.memcpyHostToDevice)
        handle = cp.cuda.runtime.ipcGetMemHandle(arr.data.mem.ptr)
    info = dict(request, handle=bytes(handle).hex(), token=token.tobytes().hex(),
                shape=arr.shape, dtype=arr.dtype.str, nbytes=arr.nbytes, pid=os.getpid())
    shared = _SharedMemory(info, device_id, memory=arr.data.mem)
    _publish(f'{key}.json', info)
    return shared


def get_shared_array(filename, exten=1, target_device_idx=None, precision=None):
    '''
    Get a shared, read-only copy of an image extension of a FITS file,
    as returned by specula.lib.fits_io.load_fits_array() with the same
    arguments. Called by load_fits_array() itself, so that sharing is
    transparent for its users.

    Parameters
    ----------
    filename: str
        FITS file name
    exten: int
        FITS extension
    target_device_idx: int
        device of the array (None for the default device)
    precision: int
        SPECULA precision (None for the global precision)

    Returns
    -------
    cupy.ndarray, or None if the array is on the CPU. If the array cannot
    be shared, it is loaded locally.
    '''
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx
    if target_device_idx < 0 or specula.cp is None:
        return None
    if precision is None:
        precision = specula.global_precision

    cp = specula.cp
    st = os.stat(filename)
    request = {'file': os.path.abspath(filename), 'exten': exten, 'precision': precision,
               'pci_bus_id': cp.cuda.Device(target_device_idx).pci_bus_id}
    key = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()
    request.update(mtime_ns=st.st_mtime_ns, size=st.st_size)

    def current(info):
        return isinstance(info, dict) and all(info.get(k) == v for k, v in request.items())

    # Each handle can be opened only once per process, and not by
    # the process that published it
    shared = _by_key.get(key)
    if shared is None or not current(shared.info):
        shared = None
        info = _read_json(f'{key}.json')
        if current(info):
            try:
                shared = _SharedMemory(info, target_device_idx)
            except Exception:
                # Stale handle: the publisher has freed the array or
                # terminated. Or the file is corrupted.
                pass
        if shared is None:
            arr = _load(request, target_device_idx)
            try:
                shared = _publish_array(key, request, arr, target_device_idx)
            except Exception as e:
                specula.get_specula_logger(__name__).warning(
                    f'Cannot share {filename} on GPU {target_device_idx}: {e}')
                return arr
        _by_key[key] = shared
        _by_ptr[(target_device_idx, shared.ptr)] = shared
    return shared.array()


def is_shared(arr):
    '''True if *arr* (or the array it is a view of) is a shared array'''
    cp = specula.cp
    if cp is None or not isinstance(arr, cp.ndarray):
        return False
    mem = arr.data.mem
    # UnownedMemory does not expose its owner: it is found by pointer,
    # it is alive as long as the memory references it
    return isinstance(mem, cp.cuda.UnownedMemory) and (mem.device_id, mem.ptr) in _by_ptr


def writable(arr):
    '''
    Return *arr*, or a private copy of it if it is a shared array.
    To be used by the data objects before writing into their arrays.
    '''
    return arr.copy() if is_shared(arr) else arr


def list_arrays():
    '''Info of the published arrays'''
    if not os.path.isdir(DIR):
        return []
    infos = [_read_json(name) for name in sorted(os.listdir(DIR))
             if name.endswith('.json') and not name.startswith('.')]
    return [info for info in infos if info is not None]


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process of another user
        return True


def main():
    rows = [info for info in list_arrays() if _pid_alive(info['pid'])]
    for info in rows:
        print(f'{info["pci_bus_id"]} {info["nbytes"] / 2**20:10.1f} MiB  pid={info["pid"]}  '
              f'{tuple(info["shape"])} {info["dtype"]} {info["file"]}[{info["exten"]}]')
    print(f'{len(rows)} arrays published by running processes, '
          f'{sum(i["nbytes"] for i in rows) / 2**30:.2f} GiB')


if __name__ == '__main__':
    sys.exit(main())
