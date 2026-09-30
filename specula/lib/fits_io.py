import numpy as np
from astropy.io import fits

import specula

_BITPIX2DTYPE = {8: np.uint8, 16: np.int16, 32: np.int32, 64: np.int64,
                 -32: np.float32, -64: np.float64}

_CHUNK_BYTES = 64 << 20

_byteswap_kernel = None


def _byteswap(x):
    '''In-place byteswap of a cupy array of unsigned integers'''
    global _byteswap_kernel
    if _byteswap_kernel is None:
        _byteswap_kernel = specula.cp.ElementwiseKernel(
            'T x', 'T y',
            '''
            T v = x;
            const unsigned char* s = reinterpret_cast<const unsigned char*>(&v);
            unsigned char* d = reinterpret_cast<unsigned char*>(&y);
            for (int i = 0; i < sizeof(T); i++)
                d[i] = s[sizeof(T) - 1 - i];
            ''',
            'specula_byteswap')
    _byteswap_kernel(x, x)


def _target_dtype(src_dtype, precision):
    '''Floating point data is cast to the SPECULA float dtype, the rest keeps its type'''
    if np.issubdtype(src_dtype, np.floating):
        if precision is None:
            precision = specula.global_precision
        return np.dtype(specula.cpu_float_dtype_list[precision])
    return src_dtype


def _readinto_full(f, buf):
    got = 0
    while got < len(buf):
        n = f.readinto(buf[got:])
        if not n:
            raise EOFError(f'Unexpected end of file reading {f.name}')
        got += n


def load_fits_array(filename, exten=1, target_device_idx=None, precision=None):
    '''
    Read the data of an image extension of a FITS file into a
    numpy or cupy array allocated on *target_device_idx*
    (same convention as BaseTimeObj: None for the default device,
    -1 for the CPU).

    Floating point data is converted to the float dtype of
    *precision* (None for the global precision), other data types
    are kept, always in native byte order.

    On GPU, the file is read in chunks into two pinned host buffers,
    which are uploaded asynchronously while the next chunk is read,
    and are byteswapped and converted on the GPU. This avoids
    full-size temporary copies in host memory.
    '''
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx

    with fits.open(filename) as hdul:
        hdu = hdul[exten]
        info = hdu.fileinfo()
        hdr = hdu.header
        fast = (target_device_idx >= 0
                and type(hdu) in (fits.PrimaryHDU, fits.ImageHDU)
                and info['file'].compression is None
                and hdr.get('BSCALE', 1) == 1 and hdr.get('BZERO', 0) == 0
                and 'BLANK' not in hdr
                and hdr.get('BITPIX') in _BITPIX2DTYPE)
        if not fast:
            data = hdu.data
            dtype = _target_dtype(data.dtype.newbyteorder('='), precision)
            if target_device_idx < 0:
                return np.array(data, dtype=dtype)
            with specula.cp.cuda.Device(target_device_idx):
                return specula.cp.asarray(np.asarray(data, dtype=dtype))
        shape = hdu.shape
        offset = info['datLoc']

    return _load_image_gpu(filename, offset, shape,
                           np.dtype(_BITPIX2DTYPE[hdr['BITPIX']]),
                           target_device_idx, precision)


def _load_image_gpu(filename, offset, shape, src_dtype, target_device_idx, precision):
    cp = specula.cp
    dtype = _target_dtype(src_dtype, precision)
    itemsize = src_dtype.itemsize
    nbytes = int(np.prod(shape)) * itemsize
    chunk = min(_CHUNK_BYTES, max(nbytes, itemsize))
    uint_dtype = np.dtype(f'u{itemsize}')

    with cp.cuda.Device(target_device_idx):
        out = cp.empty(shape, dtype=dtype)
        out_flat = out.reshape(-1)
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        pinned = [cp.cuda.alloc_pinned_memory(chunk) for _ in range(2)]
        host = [np.frombuffer(p, dtype=np.uint8, count=chunk) for p in pinned]
        events = [None, None]

        with open(filename, 'rb', buffering=0) as f:
            f.seek(offset)
            pos = 0
            i = 0
            while pos < nbytes:
                k = i % 2
                if events[k] is not None:
                    events[k].synchronize()
                n = min(chunk, nbytes - pos)
                _readinto_full(f, memoryview(host[k])[:n])
                start = pos // itemsize
                count = n // itemsize
                with streams[k]:
                    if dtype == src_dtype:
                        dst = out_flat[start:start + count].view(uint_dtype)
                    else:
                        dst = cp.empty(count, dtype=uint_dtype)
                    dst.data.copy_from_host_async(host[k].ctypes.data, n, streams[k])
                    if itemsize > 1:
                        _byteswap(dst)
                    if dtype != src_dtype:
                        out_flat[start:start + count] = dst.view(src_dtype)
                    events[k] = streams[k].record()
                pos += n
                i += 1
        for s in streams:
            s.synchronize()
    return out
