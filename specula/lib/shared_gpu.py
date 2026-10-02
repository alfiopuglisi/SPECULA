'''
Sharing of read-only GPU arrays between simulation processes.

Several simulations running on the same GPU usually load the same large
calibration arrays from FITS files (interaction and reconstruction matrices,
influence functions, modes-to-commands matrices). With this module, the
first simulation that loads an array allocates it with the CUDA virtual
memory management API (cuMemCreate), exporting it with a POSIX file
descriptor; the other simulations map the same device memory instead of
allocating and loading their own copy.

Sharing is done by specula.lib.fits_io.load_fits_array(), used by the
restore() methods of the data objects, and is transparent for its users.
There is nothing to start: simulations share the arrays they load at the
same time. Optionally, a *keeper* keeps the arrays for an idle time after
the last simulation using them terminates, so that simulations run one
after the other find them already loaded::

    python -m specula.lib.shared_gpu keep &    # optional
    python -m specula.lib.shared_gpu list      # show the shared arrays
    python -m specula.lib.shared_gpu stop      # stop the keeper

The keeper does not use CUDA: it only holds the file descriptors, so it
uses no GPU memory except the arrays it keeps.

How it works
------------
There is no connection between the processes, only files in a directory
private to the user (DIR), where <key> is a hash of file name,
modification time, size, FITS extension, precision and GPU:

- <key>.json: written by the simulation that loads the array (shape,
  dtype, allocation size and a unique id of the allocation)
- <key>.<pid>.fd: written by each process that holds a file descriptor
  of the allocation (simulations and keeper), with its number
- <key>.lock: locked with flock() while a simulation loads the array

A simulation that needs an array takes the file descriptor from any
process that holds it, with pidfd_getfd() (Linux 5.6+), and becomes a
holder itself. The CUDA driver keeps the memory alive while any process
holds a file descriptor or a mapping of it, so the array survives the
simulation that loaded it, and is freed when the last holder terminates.

Notes
-----
- Shared arrays are read-only by convention: CuPy cannot enforce it, and
  an in-place write is seen by all the simulations. The data objects
  that use shared arrays make a private copy before writing into them
  (copy-on-write), but views of them must not be modified in place
  (for example ``intmat.modes[3:5] += 1``).
- pidfd_getfd() requires the permission to ptrace the other process.
  With the Yama security module in mode 1 (Ubuntu default), each holder
  allows it with prctl(PR_SET_PTRACER_ANY); in modes 2 and 3 sharing
  is not possible.
- Arrays are loaded locally, as usual, when the array is on the CPU, the
  GPU or driver does not support VMM with POSIX file descriptors, or
  pidfd_getfd() is not allowed.
- GPUs are identified by PCI bus id, so that the processes can have
  different CUDA_VISIBLE_DEVICES.
'''

import os
import sys
import json
import time
import uuid
import errno
import fcntl
import ctypes
import select
import signal
import hashlib
import logging
import argparse
import tempfile
import threading
import weakref

import numpy as np

import specula

# Directory of the sharing files, one per user. /dev/shm is a RAM filesystem
DIR = os.path.join('/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir(),
                   f'specula_shared_gpu_{os.getuid()}')

# Maximum time a simulation waits for another one loading the same array [s]
WAIT_TIMEOUT = 600

# Interval of the keeper checks of the arrays in use [s]
CHECK_INTERVAL = 30


def _logger():
    return logging.getLogger(__name__)


def _path(name):
    return os.path.join(DIR, name)


def _write_atomic(name, data):
    '''Write a file of DIR, so that readers see either nothing or all of it'''
    tmp = _path(f'.{name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with open(tmp, 'w') as f:
        f.write(data)
    os.replace(tmp, _path(name))


def _read_json(name):
    try:
        with open(_path(name)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _remove(name):
    try:
        os.unlink(_path(name))
    except FileNotFoundError:
        pass


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _holders(key, alloc_id):
    '''
    {pid: fd} of the live processes that hold a file descriptor of the
    allocation *alloc_id* of *key*. Removes the files of terminated processes.
    '''
    holders = {}
    prefix = key + '.'
    for name in os.listdir(DIR):
        if not (name.startswith(prefix) and name.endswith('.fd')):
            continue
        pid = name[len(prefix):-len('.fd')]
        if not pid.isdigit():
            continue
        if not _pid_alive(int(pid)):
            _remove(name)
            continue
        entry = _read_json(name)
        if entry is not None and entry['id'] == alloc_id:
            holders[int(pid)] = entry['fd']
    return holders


def _remove_unused(key, alloc_id):
    '''
    Remove the files of *key* if no process holds the allocation *alloc_id*
    and no process is loading it
    '''
    if _holders(key, alloc_id):
        return
    info = _read_json(f'{key}.json')
    if info is not None and info['id'] == alloc_id:
        _remove(f'{key}.json')
    _remove_lock(key)


def _remove_lock(key):
    '''Remove the lock file of *key*, unless a process is loading the array'''
    try:
        lock_fd = os.open(_path(f'{key}.lock'), os.O_RDWR)
    except FileNotFoundError:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _remove(f'{key}.lock')
    except BlockingIOError:
        pass
    finally:
        os.close(lock_fd)


# ---------------------------------------------------------------------------
# File descriptors of other processes

_libc = ctypes.CDLL(None, use_errno=True)
_SYS_pidfd_open = 434      # same numbers on all Linux architectures
_SYS_pidfd_getfd = 438
_ptracer_allowed = False


def _pidfd_getfd(pid, fd):
    '''Duplicate the file descriptor *fd* of process *pid* into this process'''
    pidfd = _libc.syscall(_SYS_pidfd_open, pid, 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    try:
        new_fd = _libc.syscall(_SYS_pidfd_getfd, pidfd, fd, 0)
        if new_fd < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_getfd')
        return new_fd
    finally:
        os.close(pidfd)


def _allow_ptracer():
    '''
    Allow the other processes of the user to call pidfd_getfd() on this
    one, also with Yama in mode 1. Fails harmlessly without Yama.
    '''
    global _ptracer_allowed
    if not _ptracer_allowed:
        PR_SET_PTRACER = 0x59616d61
        PR_SET_PTRACER_ANY = ctypes.c_ulong(-1)
        _libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
        _ptracer_allowed = True


def _publish_fd(key, alloc_id, fd):
    _allow_ptracer()
    _write_atomic(f'{key}.{os.getpid()}.fd', json.dumps({'id': alloc_id, 'fd': fd}))


# ---------------------------------------------------------------------------
# CUDA driver API (virtual memory management) with ctypes

class _CUmemLocation(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [('compressionType', ctypes.c_ubyte), ('gpuDirectRDMACapable', ctypes.c_ubyte),
                ('usage', ctypes.c_ushort), ('reserved', ctypes.c_ubyte * 4)]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('requestedHandleTypes', ctypes.c_int),
                ('location', _CUmemLocation), ('win32HandleMetaData', ctypes.c_void_p),
                ('allocFlags', _CUmemAllocFlags)]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [('location', _CUmemLocation), ('flags', ctypes.c_int)]


_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
_CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103

_cuda_lib = None
_supported = {}
_granularity = {}


class CudaDriverError(RuntimeError):
    pass


def _cuda():
    global _cuda_lib
    if _cuda_lib is None:
        _cuda_lib = ctypes.CDLL('libcuda.so.1')
        _cuda_lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        _check(_cuda_lib.cuInit(0), 'cuInit')
    return _cuda_lib


def _check(err, what):
    if err != 0:
        name = ctypes.c_char_p()
        _cuda_lib.cuGetErrorName(err, ctypes.byref(name))
        raise CudaDriverError(f'{what}: {name.value.decode() if name.value else err}')


def _prop(device_id):
    prop = _CUmemAllocationProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    return prop


def _vmm_supported(device_id):
    '''True if the device supports VMM allocations shared with POSIX file descriptors'''
    if device_id not in _supported:
        try:
            cuda = _cuda()
            dev = ctypes.c_int()
            _check(cuda.cuDeviceGet(ctypes.byref(dev), device_id), 'cuDeviceGet')
            ok = True
            for attr in (_CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                         _CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED):
                value = ctypes.c_int()
                _check(cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, dev), 'cuDeviceGetAttribute')
                ok = ok and value.value == 1
            _supported[device_id] = ok and sys.platform.startswith('linux')
        except (OSError, CudaDriverError):
            _supported[device_id] = False
    return _supported[device_id]


def _alloc_size(device_id, nbytes):
    '''*nbytes* rounded up to the allocation granularity'''
    if device_id not in _granularity:
        value = ctypes.c_size_t()
        _check(_cuda().cuMemGetAllocationGranularity(ctypes.byref(value), ctypes.byref(_prop(device_id)), 0),
               'cuMemGetAllocationGranularity')
        _granularity[device_id] = value.value
    g = _granularity[device_id]
    return max(1, -(-nbytes // g)) * g


# ---------------------------------------------------------------------------
# Simulation side

# Reentrant, since _VmmMapping.__del__() can be called by the garbage collector
_lock = threading.RLock()
_mappings = weakref.WeakValueDictionary()          # key -> _VmmMapping
_mappings_by_ptr = weakref.WeakValueDictionary()   # (device_id, ptr) -> _VmmMapping
_warned = set()


class _VmmMapping:
    '''
    A VMM allocation mapped in this process. A new allocation if *fd*
    is None, otherwise the one shared by *fd*, which then belongs to the
    mapping. It is the owner of the UnownedMemory of the arrays built on
    it, so it is unmapped and released when the last array or view
    referencing it is garbage collected.
    '''
    def __init__(self, device_id, size, fd=None):
        self.device_id = device_id
        self.size = size
        self.fd = fd
        self.handle = None
        self.ptr = None
        self.published = None
        self.info = None
        self.mapped = False
        cuda = _cuda()
        cp = specula.cp
        with cp.cuda.Device(device_id):
            try:
                cp.cuda.runtime.free(0)   # make sure that the context is current
                handle = ctypes.c_uint64()
                if fd is None:
                    _check(cuda.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size),
                                            ctypes.byref(_prop(device_id)), ctypes.c_uint64(0)),
                           'cuMemCreate')
                else:
                    _check(cuda.cuMemImportFromShareableHandle(
                        ctypes.byref(handle), ctypes.c_void_p(fd),
                        _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), 'cuMemImportFromShareableHandle')
                self.handle = handle.value
                ptr = ctypes.c_uint64()
                _check(cuda.cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(0),
                                                ctypes.c_uint64(0), ctypes.c_uint64(0)), 'cuMemAddressReserve')
                self.ptr = ptr.value
                _check(cuda.cuMemMap(ctypes.c_uint64(self.ptr), ctypes.c_size_t(size), ctypes.c_size_t(0),
                                     ctypes.c_uint64(self.handle), ctypes.c_uint64(0)), 'cuMemMap')
                self.mapped = True
                access = _CUmemAccessDesc()
                access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = device_id
                access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check(cuda.cuMemSetAccess(ctypes.c_uint64(self.ptr), ctypes.c_size_t(size),
                                           ctypes.byref(access), ctypes.c_size_t(1)), 'cuMemSetAccess')
            except Exception:
                self._release()
                raise
        _mappings_by_ptr[(device_id, self.ptr)] = self

    def export(self):
        '''Export the allocation to a file descriptor, owned by the mapping'''
        if self.fd is None:
            fd = ctypes.c_int()
            _check(_cuda().cuMemExportToShareableHandle(ctypes.byref(fd), ctypes.c_uint64(self.handle),
                                                        _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                                        ctypes.c_uint64(0)), 'cuMemExportToShareableHandle')
            self.fd = fd.value
        return self.fd

    def publish(self, key, info):
        '''Tell the other processes that this one holds the allocation of *key*'''
        self.info = info
        _publish_fd(key, info['id'], self.export())
        self.published = key

    def array(self):
        cp = specula.cp
        mem = cp.cuda.UnownedMemory(self.ptr, self.info['nbytes'], self, device_id=self.device_id)
        return cp.ndarray(tuple(self.info['shape']), dtype=np.dtype(self.info['dtype']),
                          memptr=cp.cuda.MemoryPointer(mem, 0))

    def _release(self):
        cuda = _cuda_lib
        if self.published is not None:
            _remove(f'{self.published}.{os.getpid()}.fd')
            _remove_unused(self.published, self.info['id'])
            self.published = None
        if self.mapped:
            cuda.cuMemUnmap(ctypes.c_uint64(self.ptr), ctypes.c_size_t(self.size))
            self.mapped = False
        if self.ptr is not None:
            cuda.cuMemAddressFree(ctypes.c_uint64(self.ptr), ctypes.c_size_t(self.size))
            self.ptr = None
        if self.handle is not None:
            cuda.cuMemRelease(ctypes.c_uint64(self.handle))
            self.handle = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        try:
            cp = specula.cp
            # The garbage collector can run during a CUDA graph capture, where
            # synchronizing is not allowed: the mapping is then left as it is
            # until the process terminates
            if cp.cuda.get_current_stream().is_capturing():
                return
            with cp.cuda.Device(self.device_id):
                # Work queued on the memory must be completed before unmapping it
                cp.cuda.runtime.deviceSynchronize()
                self._release()
        except Exception:
            # Interpreter shutdown: the driver cleans up anyway
            pass


def _attach(key, info, device_id):
    '''Map the allocation of *key* from any process that holds it. Returns the mapping, or None'''
    for pid, fd in _holders(key, info['id']).items():
        if pid == os.getpid():
            continue
        try:
            new_fd = _pidfd_getfd(pid, fd)
        except OSError as e:
            if e.errno == errno.EPERM:
                raise
            continue    # the process has terminated in the meantime
        try:
            mapping = _VmmMapping(device_id, info['size'], fd=new_fd)
        except CudaDriverError:
            continue    # PID reused by another process
        mapping.publish(key, info)
        return mapping
    return None


def _load(key, request, device_id):
    '''Load the array into a new VMM allocation, and publish it'''
    from specula.lib.fits_io import load_fits_array
    cp = specula.cp
    created = weakref.WeakValueDictionary()

    def alloc(size):
        mapping = _VmmMapping(device_id, _alloc_size(device_id, size))
        created[mapping.ptr] = mapping
        return cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(mapping.ptr, size, mapping,
                                                           device_id=device_id), 0)

    # All the allocations of the loading are VMM ones: the result owns a whole allocation
    with cp.cuda.Device(device_id), cp.cuda.using_allocator(alloc):
        arr = load_fits_array(request['file'], request['exten'], device_id,
                              request['precision'], shared=False)
        if arr.data.ptr not in created or not arr.flags.c_contiguous:
            arr = arr.copy()
        cp.cuda.runtime.deviceSynchronize()
    mapping = created[arr.data.ptr]
    info = dict(request, id=uuid.uuid4().hex, shape=arr.shape, dtype=arr.dtype.str,
                nbytes=arr.nbytes, size=mapping.size, loaded_by=os.getpid())
    # The fd file first: whoever finds the json finds a holder
    mapping.publish(key, info)
    _write_atomic(f'{key}.json', json.dumps(info))
    _logger().info(f'Loaded {request["file"]} into shared GPU memory: {arr.shape} {arr.dtype}, '
                   f'{arr.nbytes / 2**20:.1f} MiB')
    return mapping


def _attach_or_load(key, request, device_id):
    os.makedirs(DIR, mode=0o700, exist_ok=True)
    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        info = _read_json(f'{key}.json')
        if info is not None:
            mapping = _attach(key, info, device_id)
            if mapping is not None:
                return mapping
            # Nobody holds it anymore: load it again

        lock_fd = os.open(_path(f'{key}.lock'), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.05)     # another simulation is loading it
                continue
            # It may have been published after the check above
            info = _read_json(f'{key}.json')
            if info is not None:
                mapping = _attach(key, info, device_id)
                if mapping is not None:
                    return mapping
            return _load(key, request, device_id)
        finally:
            os.close(lock_fd)        # also releases the lock
    _logger().warning(f'Timeout waiting for another process loading {request["file"]}: loaded locally')
    return None


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
    cupy.ndarray, or None if the array cannot be shared (see the module
    documentation). In this case the caller must load the array itself.
    '''
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx
    if target_device_idx < 0 or specula.cp is None or not _vmm_supported(target_device_idx):
        return None
    if precision is None:
        precision = specula.global_precision

    cp = specula.cp
    st = os.stat(filename)
    request = {'file': os.path.abspath(filename), 'mtime_ns': st.st_mtime_ns, 'size': st.st_size,
               'exten': exten, 'precision': precision,
               'pci_bus_id': cp.cuda.Device(target_device_idx).pci_bus_id}
    key = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()

    with _lock:
        mapping = _mappings.get(key)
        if mapping is None:
            try:
                mapping = _attach_or_load(key, request, target_device_idx)
            except OSError as e:
                if e.errno != errno.EPERM:
                    raise
                if 'pidfd' not in _warned:
                    _warned.add('pidfd')
                    _logger().warning('pidfd_getfd() is not allowed (see kernel.yama.ptrace_scope): '
                                      'GPU arrays are not shared')
                return None
            if mapping is None:
                return None
            _mappings[key] = mapping
    return mapping.array()


def is_shared(arr):
    '''True if *arr* (or the array it is a view of) is in shared memory'''
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


def list_arrays():
    '''Info of the shared arrays, with the PIDs of the processes that hold them'''
    if not os.path.isdir(DIR):
        return []
    infos = []
    for name in sorted(os.listdir(DIR)):
        if name.endswith('.json') and not name.startswith('.'):
            info = _read_json(name)
            if info is not None:
                holders = _holders(name[:-len('.json')], info['id'])
                if holders:
                    infos.append(dict(info, holders=sorted(holders)))
    return infos


# ---------------------------------------------------------------------------
# Keeper

_IN_CLOSE_WRITE = 0x08
_IN_MOVED_TO = 0x80


class _DirWatcher:
    '''Waits until a file is written in a directory: inotify on Linux, polling elsewhere'''

    def __init__(self, path):
        self.fd = None
        try:
            fd = _libc.inotify_init1(os.O_CLOEXEC)
            if fd >= 0 and _libc.inotify_add_watch(fd, path.encode(),
                                                   _IN_CLOSE_WRITE | _IN_MOVED_TO) >= 0:
                self.fd = fd
        except AttributeError:
            pass

    def wait(self, timeout):
        if self.fd is None:
            time.sleep(min(0.2, timeout))
        elif select.select([self.fd], [], [], timeout)[0]:
            # The events are not needed: the caller scans the directory again
            os.read(self.fd, 65536)


def keeper_pid():
    '''PID of the running keeper, or None'''
    try:
        with open(_path('keeper.pid')) as f:
            pid = int(f.read())
        return pid if _pid_alive(pid) else None
    except (OSError, ValueError):
        return None


class Keeper:
    '''
    Holds the file descriptors of the shared arrays, and releases them
    when no simulation has used them for *idle_timeout* seconds.
    It does not use CUDA.
    '''

    def __init__(self, idle_timeout=600):
        self.idle_timeout = idle_timeout
        self.held = {}    # key -> [alloc_id, fd, last_used]

    def _release(self, key, reason):
        alloc_id, fd, _ = self.held.pop(key)
        info = _read_json(f'{key}.json')
        _remove(f'{key}.{os.getpid()}.fd')
        os.close(fd)
        _remove_unused(key, alloc_id)
        _logger().info(f'Released {info["file"] if info else key} ({reason})')

    def check(self):
        now = time.monotonic()
        published = {}
        for name in os.listdir(DIR):
            if name.endswith('.json') and not name.startswith('.'):
                info = _read_json(name)
                if info is not None:
                    published[name[:-len('.json')]] = info
        for key in list(self.held):
            if key not in published or published[key]['id'] != self.held[key][0]:
                self._release(key, 'replaced')
        # Locks of the arrays that are not loaded anymore
        for name in os.listdir(DIR):
            if name.endswith('.lock') and name[:-len('.lock')] not in published:
                _remove_lock(name[:-len('.lock')])
        for key, info in published.items():
            users = {pid: fd for pid, fd in _holders(key, info['id']).items() if pid != os.getpid()}
            if key not in self.held:
                for pid, fd in users.items():
                    try:
                        own_fd = _pidfd_getfd(pid, fd)
                    except OSError:
                        continue
                    self.held[key] = [info['id'], own_fd, now]
                    _publish_fd(key, info['id'], own_fd)
                    _logger().info(f'Keeping {info["file"]} ({info["nbytes"] / 2**20:.1f} MiB)')
                    break
                else:
                    if not users:
                        # Stale: all its holders have terminated
                        _remove_unused(key, info['id'])
                continue
            if users:
                self.held[key][2] = now
            elif now - self.held[key][2] >= self.idle_timeout:
                self._release(key, f'unused for {self.idle_timeout / 60:g} minutes')

    def run(self):
        os.makedirs(DIR, mode=0o700, exist_ok=True)
        os.chmod(DIR, 0o700)
        pid = keeper_pid()
        if pid is not None:
            raise RuntimeError(f'A keeper is already running (PID {pid})')
        # SIGTERM (stop command) terminates like Ctrl-C
        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
        watcher = _DirWatcher(DIR)
        _write_atomic('keeper.pid', str(os.getpid()))
        _logger().info(f'Shared GPU array keeper running, files in {DIR}'
                       + ('' if watcher.fd is not None else ' (polling, inotify not available)'))
        try:
            while True:
                self.check()
                watcher.wait(min(CHECK_INTERVAL, self.idle_timeout) if self.held else None)
        except KeyboardInterrupt:
            pass
        finally:
            # The arrays still used by simulations stay alive
            for key in list(self.held):
                self._release(key, 'keeper stopped')
            _remove('keeper.pid')
            _logger().info('Shared GPU array keeper stopped')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m specula.lib.shared_gpu',
                                     description='GPU arrays shared between SPECULA simulations')
    parser.add_argument('command', choices=['keep', 'list', 'stop'])
    parser.add_argument('--idle-timeout', type=float, default=10,
                        help='keep: release the arrays not used by any simulation '
                             'for this time [minutes] (default: 10)')
    args = parser.parse_args(argv)

    if args.command == 'keep':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
        Keeper(args.idle_timeout * 60).run()
    elif args.command == 'list':
        keeper = keeper_pid()
        rows = list_arrays()
        for info in rows:
            sims = [pid for pid in info['holders'] if pid != keeper]
            kept = ' kept' if keeper in info['holders'] else ''
            print(f'{info["pci_bus_id"]} {info["nbytes"] / 2**20:10.1f} MiB  '
                  f'simulations={len(sims)}{kept}  '
                  f'{tuple(info["shape"])} {info["dtype"]} {info["file"]}[{info["exten"]}]')
        print(f'{len(rows)} arrays, {sum(i["nbytes"] for i in rows) / 2**30:.2f} GiB'
              + ('' if keeper else ' (no keeper running)'))
    elif args.command == 'stop':
        pid = keeper_pid()
        if pid is None:
            raise SystemExit('The keeper is not running')
        os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    sys.exit(main())
