'''
Sharing of read-only GPU arrays between simulation processes (CUDA IPC).

Several simulations running on the same GPU usually load the same large
calibration arrays from FITS files (interaction and reconstruction matrices,
influence functions, modes-to-commands matrices). With this module, a single
*holder* process loads each array once and exports it with a CUDA IPC
handle; the simulations map the same device memory instead of allocating
and loading their own copy.

Usage::

    # start the holder (once per machine and user, it serves all GPUs)
    python -m specula.lib.shared_gpu serve &

    # the simulations started while the holder runs share their arrays
    specula params.yml ...

    python -m specula.lib.shared_gpu list    # show the loaded arrays
    python -m specula.lib.shared_gpu stop    # stop the holder

Sharing is done by specula.lib.fits_io.load_fits_array(), used by the
restore() methods of the data objects, and is transparent for its users.
When the holder is not running, or the array is on the CPU, everything
is loaded locally as usual.

There is no connection between the holder and the simulations, only
files in a directory private to the user (DIR):

- holder.pid: the PID of the running holder
- <key>.request: written by a simulation that needs an array (JSON
  with file name, modification time, size, FITS extension, precision
  and GPU)
- <key>.json: written by the holder when the array is loaded (JSON with
  the IPC handle, shape and dtype), or <key>.error if it cannot load it
- <key>.<pid>.user: written by each simulation that maps the array, and
  removed when it unmaps it. The holder ignores the files of processes
  that have terminated.

<key> is a hash of the request, so a simulation finds the array of a file
from the file name alone. The holder is woken up by the requests with
inotify (on Linux, otherwise it checks the directory periodically).

Notes
-----
- Shared arrays are read-only by convention: CuPy cannot enforce it, and
  an in-place write is seen by all the simulations. The data objects
  that use shared arrays make a private copy before writing into them
  (copy-on-write), but views of them must not be modified in place
  (for example ``intmat.modes[3:5] += 1``).
- The CUDA driver keeps the device memory alive while any process maps
  it: the holder can free its arrays or terminate at any time without
  affecting the running simulations. Only the simulations started after
  that load the arrays locally, since a handle cannot be opened after
  its memory has been freed by the holder.
- The holder frees an array when no simulation has used it for an idle
  time (10 minutes by default, ``serve --idle-timeout``), so that
  simulations run one after the other find it already loaded. A file
  that is rewritten is loaded again, and the older version is freed.
- GPUs are identified by PCI bus id, so that the holder and the
  simulations can have different CUDA_VISIBLE_DEVICES.
'''

import os
import sys
import json
import time
import ctypes
import select
import signal
import hashlib
import argparse
import tempfile
import threading
import weakref

import numpy as np

import specula

# Directory of the holder files, one per user. /dev/shm is a RAM filesystem
DIR = os.path.join('/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir(),
                   f'specula_shared_gpu_{os.getuid()}')

# Maximum time a simulation waits for the holder to load an array [s]
WAIT_TIMEOUT = 600

# Interval of the holder checks of the arrays in use [s]
CHECK_INTERVAL = 30


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


def holder_pid():
    '''PID of the running holder, or None'''
    try:
        with open(_path('holder.pid')) as f:
            pid = int(f.read())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Simulation side

# Reentrant, since _IpcMapping.__del__() can be called by the garbage collector
_lock = threading.RLock()
_mappings = weakref.WeakValueDictionary()          # key -> _IpcMapping
_mappings_by_ptr = weakref.WeakValueDictionary()   # (device_id, ptr) -> _IpcMapping


class _IpcMapping:
    '''
    A device memory region opened from an IPC handle.
    It is the owner of the UnownedMemory of the arrays built on it,
    so it is closed when the last array or view referencing it is
    garbage collected.
    '''
    def __init__(self, key, info, device_id):
        self.user_file = f'{key}.{os.getpid()}.user'
        self.shape = tuple(info['shape'])
        self.dtype = np.dtype(info['dtype'])
        self.nbytes = info['nbytes']
        self.device_id = device_id
        cp = specula.cp
        with cp.cuda.Device(device_id):
            self.ptr = cp.cuda.runtime.ipcOpenMemHandle(bytes.fromhex(info['handle']))
        # Tell the holder that the array is in use
        open(_path(self.user_file), 'w').close()

    def __del__(self):
        try:
            cp = specula.cp
            # The garbage collector can run during a CUDA graph capture, where
            # synchronizing is not allowed: the mapping is then left open
            # until the process terminates
            if cp.cuda.get_current_stream().is_capturing():
                return
            with cp.cuda.Device(self.device_id):
                # Work queued on the memory must be completed before unmapping it
                cp.cuda.runtime.deviceSynchronize()
                cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
            _remove(self.user_file)
        except Exception:
            # Interpreter shutdown: the driver cleans up anyway
            pass


def _wait_for_holder(key, request):
    '''Ask the holder to load an array, and wait for it. Returns its info, or None'''
    _write_atomic(f'{key}.request', json.dumps(request))
    t0 = time.monotonic()
    while time.monotonic() - t0 < WAIT_TIMEOUT:
        info = _read_json(f'{key}.json')
        if info is not None:
            return info
        if os.path.exists(_path(f'{key}.error')) or holder_pid() is None:
            return None
        time.sleep(0.05)
    specula.get_specula_logger(__name__).warning(
        f'Timeout waiting for the shared GPU array holder: {request["file"]} loaded locally')
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
    cupy.ndarray, or None if the holder is not running, it cannot load
    the array, or the array is on the CPU. In this case the caller must
    load the array itself.
    '''
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx
    if target_device_idx < 0 or specula.cp is None or holder_pid() is None:
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
            info = _read_json(f'{key}.json')
            if info is None:
                info = _wait_for_holder(key, request)
                if info is None:
                    return None
            try:
                mapping = _IpcMapping(key, info, target_device_idx)
            except cp.cuda.runtime.CUDARuntimeError:
                # Stale handle: the holder has terminated, or freed the array
                return None
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


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _users():
    '''Live users of each array, as {key: [pid, ...]}. Removes the files of terminated processes'''
    users = {}
    for name in os.listdir(DIR):
        parts = name.split('.')
        if len(parts) == 3 and parts[2] == 'user' and parts[1].isdigit():
            if _pid_alive(int(parts[1])):
                users.setdefault(parts[0], []).append(int(parts[1]))
            else:
                _remove(name)
    return users


def list_arrays():
    '''Info of the arrays published by the holder, with their live users'''
    if not os.path.isdir(DIR):
        return []
    users = _users()
    infos = []
    for name in sorted(os.listdir(DIR)):
        if name.endswith('.json') and not name.startswith('.'):
            info = _read_json(name)
            if info is not None:
                infos.append(dict(info, users=users.get(name[:-len('.json')], [])))
    return infos


# ---------------------------------------------------------------------------
# Holder side

_IN_CLOSE_WRITE = 0x08
_IN_MOVED_TO = 0x80


class _DirWatcher:
    '''Waits until a file is written in a directory: inotify on Linux, polling elsewhere'''

    def __init__(self, path):
        self.fd = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.inotify_init1(os.O_CLOEXEC)
            if fd >= 0 and libc.inotify_add_watch(fd, path.encode(),
                                                  _IN_CLOSE_WRITE | _IN_MOVED_TO) >= 0:
                self.fd = fd
        except (OSError, AttributeError):
            pass

    def wait(self, timeout):
        if self.fd is None:
            time.sleep(min(0.2, timeout))
        elif select.select([self.fd], [], [], timeout)[0]:
            # The events are not needed: the caller scans the directory again
            os.read(self.fd, 65536)


def _load_fits(filename, exten, device_idx, precision):
    '''
    Load the array with dedicated cudaMalloc() allocations instead of
    sub-allocations of the CuPy memory pool, so that it owns a whole
    allocation that can be exported with a IPC handle of its own.
    '''
    from specula.lib.fits_io import load_fits_array
    cp = specula.cp

    def alloc(size):
        return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)

    with cp.cuda.Device(device_idx), cp.cuda.using_allocator(alloc):
        arr = load_fits_array(filename, exten, device_idx, precision, shared=False)
        if arr.data.ptr != arr.data.mem.ptr or not arr.flags.c_contiguous:
            arr = arr.copy()
        cp.cuda.runtime.deviceSynchronize()
    return arr


class Holder:
    '''Loads the requested arrays and keeps them for the simulations'''

    def __init__(self, logger, idle_timeout=600):
        self.logger = logger
        self.idle_timeout = idle_timeout
        self.arrays = {}      # key -> (request, array)
        self.last_used = {}   # key -> time.monotonic() of the last check with users

    def _free(self, key, reason):
        self.logger.info(f'Freeing {self.arrays[key][0]["file"]} ({reason})')
        # The handle first, so that no simulation tries to open a freed array.
        # The simulations that use the array are not affected, the driver
        # keeps the memory they map.
        _remove(f'{key}.json')
        del self.arrays[key]
        del self.last_used[key]

    def _free_unused(self):
        now = time.monotonic()
        users = _users()
        for key in list(self.arrays):
            if key in users:
                self.last_used[key] = now
            elif now - self.last_used[key] >= self.idle_timeout:
                self._free(key, f'unused for {self.idle_timeout / 60:g} minutes')

    def _serve_request(self, key):
        request = _read_json(f'{key}.request')
        _remove(f'{key}.request')
        if request is None or key in self.arrays:
            return
        filename = request['file']
        try:
            st = os.stat(filename)
            if (st.st_mtime_ns, st.st_size) != (request['mtime_ns'], request['size']):
                raise ValueError(f'{filename} has changed')
            cp = specula.cp
            device_idx = cp.cuda.runtime.deviceGetByPCIBusId(request['pci_bus_id'])
            self.logger.info(f'Loading {filename} extension {request["exten"]} '
                             f'on GPU {request["pci_bus_id"]}')
            arr = _load_fits(filename, request['exten'], device_idx, request['precision'])
            info = dict(request, handle=cp.cuda.runtime.ipcGetMemHandle(arr.data.ptr).hex(),
                        shape=arr.shape, dtype=arr.dtype.str, nbytes=arr.nbytes)
        except Exception as e:
            self.logger.exception(f'Cannot load {filename}')
            _write_atomic(f'{key}.error', f'{type(e).__name__}: {e}')
            return

        # Free the older versions of a rewritten file
        same = ('file', 'exten', 'precision', 'pci_bus_id')
        for old_key, (old_request, _) in list(self.arrays.items()):
            if all(old_request[k] == request[k] for k in same):
                self._free(old_key, 'file rewritten')

        self.arrays[key] = (request, arr)
        self.last_used[key] = time.monotonic()
        _write_atomic(f'{key}.json', json.dumps(info))
        self.logger.info(f'Loaded {filename}: {arr.shape} {arr.dtype}, '
                         f'{arr.nbytes / 2**20:.1f} MiB')

    def serve(self):
        os.makedirs(DIR, mode=0o700, exist_ok=True)
        os.chmod(DIR, 0o700)
        pid = holder_pid()
        if pid is not None:
            raise RuntimeError(f'A holder is already running (PID {pid})')
        # Files of a holder that did not terminate cleanly
        for name in os.listdir(DIR):
            if not name.endswith('.request'):
                _remove(name)

        # SIGTERM (stop command) terminates like Ctrl-C, removing the files
        signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
        watcher = _DirWatcher(DIR)
        _write_atomic('holder.pid', str(os.getpid()))
        self.logger.info(f'Shared GPU array holder running, files in {DIR}'
                         + ('' if watcher.fd is not None else ' (polling, inotify not available)'))
        try:
            while True:
                for name in sorted(os.listdir(DIR)):
                    if name.endswith('.request') and not name.startswith('.'):
                        self._serve_request(name[:-len('.request')])
                self._free_unused()
                watcher.wait(min(CHECK_INTERVAL, self.idle_timeout) if self.arrays else None)
        except KeyboardInterrupt:
            pass
        finally:
            # Handles first, so that no simulation tries to open a freed array
            for name in os.listdir(DIR):
                if name != 'holder.pid':
                    _remove(name)
            _remove('holder.pid')
            self.logger.info('Shared GPU array holder stopped')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m specula.lib.shared_gpu',
                                     description='Holder of GPU arrays shared between '
                                                 'SPECULA simulations')
    parser.add_argument('command', choices=['serve', 'list', 'stop'])
    parser.add_argument('--idle-timeout', type=float, default=10,
                        help='serve: free the arrays not used by any simulation '
                             'for this time [minutes] (default: 10)')
    args = parser.parse_args(argv)

    if args.command == 'serve':
        # The holder serves all GPUs, with explicit device and precision in each request
        specula.init(0, precision=1)
        Holder(specula.get_specula_logger(__name__), args.idle_timeout * 60).serve()
    elif args.command == 'list':
        rows = list_arrays()
        for info in rows:
            print(f'{info["pci_bus_id"]} {info["nbytes"] / 2**20:10.1f} MiB  '
                  f'users={len(info["users"])}  '
                  f'{tuple(info["shape"])} {info["dtype"]} {info["file"]}[{info["exten"]}]')
        print(f'{len(rows)} arrays, {sum(i["nbytes"] for i in rows) / 2**30:.2f} GiB')
    elif args.command == 'stop':
        pid = holder_pid()
        if pid is None:
            raise SystemExit('The holder is not running')
        os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    sys.exit(main())
