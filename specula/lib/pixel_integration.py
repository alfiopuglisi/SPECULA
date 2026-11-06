"""
Exact integration of a band-limited periodic intensity over detector pixels.

The intensity ``|DTFT(E)|^2`` of a field E with n samples is band limited: it has
frequencies up to n - 1 per period T (T = lambda / pixel pitch of the field).
Its samples on an FFT grid with N >= 2n - 1 points per period therefore give
its Fourier coefficients exactly, and its integral over any interval is a
weighted sum of the samples. This replaces cutting and rebinning a finer FFT
when a detector has pixels that are not a multiple of the FFT sampling.
"""
import numpy as np

from specula.lib.graph_matmul import left_mul, dot_rows


def pixel_integration_matrix(n_fft, period, pxscale, npx, stop=None):
    """
    Real (npx, n_fft) matrix G that integrates a band-limited periodic
    intensity over the detector pixels, from its samples on the FFT grid.

    I(theta) has period `period` [arcsec] and frequencies up to
    n_fft / 2 - 1 per period (true for ``|FFT|^2`` of a field with
    n_fft / 2 pixels zero-padded to n_fft). Its samples I_j at
    theta_j = j * period / n_fft (FFT order, axis at j = 0) give its
    Fourier coefficients exactly, c = ifft(I), with
    I(theta) = sum_k c_k exp(-2 pi i k theta / period). The integral over
    [a, b] is then sum_k c_k B_k, with B_k the integral of the exponential,
    and, along one axis, sum_j G_j I_j with G = ifft(B). In 2D the detector
    image is G @ I @ G.T. Pixel i spans [(i - npx/2), (i + 1 - npx/2)] * pxscale,
    clipped to [-stop/2, stop/2] if a square field stop is given.
    """
    k = np.fft.fftfreq(n_fft) * n_fft
    a = (np.arange(npx) - npx / 2) * pxscale
    b = a + pxscale
    if stop is not None:
        a = np.clip(a, -stop / 2, stop / 2)
        b = np.clip(b, -stop / 2, stop / 2)
    w = -2 * np.pi * k / period
    bm = np.empty((npx, n_fft), dtype=complex)
    nz = k != 0
    bm[:, nz] = (np.exp(1j * np.outer(b, w[nz])) - np.exp(1j * np.outer(a, w[nz]))) / (1j * w[nz])
    bm[:, ~nz] = (b - a)[:, None]
    if n_fft % 2 == 0:
        # Zero by the band limit: dropping it makes G real
        bm[:, n_fft // 2] = 0
    g = np.fft.ifft(bm, axis=1)
    return g.real


def integrate_pixels(g, intensity, tmp, work, xp):
    """
    Integrate a batch of 2D intensity samples over square detector pixels,
    g @ intensity[b] @ g.T for each b, with g from pixel_integration_matrix().

    Parameters
    ----------
    g : array (npx, N)
        Pixel integration matrix (the same along both axes).
    intensity : array (nb, N, N)
        Intensity samples on the FFT grid (FFT order along both axes).
    tmp : array (nb, npx, N)
        Work buffer, contiguous.
    work : array (nb, npx, npx)
        Work buffer, contiguous.
    xp : module
        numpy or cupy.

    Returns
    -------
    image : array (nb, npx, npx)
        A view on `work` (transposed): copy it to keep it beyond the next call.
    """
    # tmp[b] = g @ intensity[b]; work[b] = g @ tmp[b].T = (tmp[b] @ g.T).T
    left_mul(g, intensity, tmp, xp)
    left_mul(g, tmp.swapaxes(1, 2), work, xp)
    return work.swapaxes(1, 2)


def fourier_integration_matrices(g, n_fft):
    """
    Real matrices (A, R) for integrate_pixels_fourier(): with them, the pixel
    integration g @ psf @ g.T of psf = irfft2(X, s=(n_fft, n_fft), norm='forward')
    is computed directly from the half spectrum X (n_fft, n_fft // 2 + 1),
    without the inverse FFT.

    psf[y, x] = sum_{ky, kx} X[ky, kx] exp(2 pi i (ky y + kx x) / n_fft) over the
    full (Hermitian) spectrum, so g @ psf @ g.T = Re(P @ X @ (w Q).T), with
    P = g @ exp(2 pi i k y / n_fft) (npx, n_fft), Q its first n_fft // 2 + 1
    columns, and w = 2 for the half-spectrum columns that stand for two (1 for
    kx = 0 and n_fft / 2, for which Q is real). In real arithmetic:

    - A (2 npx, n_fft) holds the real and imaginary parts of P, interleaved by
      rows, applied to X viewed as real numbers (n_fft, 2 M): its rows a give
      [Re(P X)[a], Im(P X)[a]] interleaved by columns;
    - R (npx, 4 M) gives the real part of the product with (w Q).T, from the
      rows 2a and 2a + 1 of A @ X, which are contiguous (4 M values).

    Parameters
    ----------
    g : array (npx, n_fft)
        Pixel integration matrix (from pixel_integration_matrix(), rolled if needed).
    n_fft : int
        FFT size, even.

    Returns
    -------
    a, r : numpy float64 arrays (2 npx, n_fft) and (npx, 4 (n_fft // 2 + 1))
    """
    if n_fft % 2:
        raise ValueError(f'n_fft must be even, got {n_fft}')
    g = np.asarray(g, dtype=float)
    npx = g.shape[0]
    m = n_fft // 2 + 1
    k = np.arange(n_fft)
    p = g @ np.exp(2j * np.pi * np.outer(k, k) / n_fft)
    w = np.full(m, 2.0)
    w[0] = w[-1] = 1.0
    q = p[:, :m] * w
    a = np.empty((2 * npx, n_fft))
    a[0::2] = p.real
    a[1::2] = p.imag
    # Re((Re PX + i Im PX)(Qr + i Qi)) = Re PX Qr - Im PX Qi, with
    # Re PX = Pr Xr - Pi Xi and Im PX = Pr Xi + Pi Xr
    r = np.empty((npx, 4 * m))
    r[:, 0:2 * m:2] = q.real        # Pr Xr
    r[:, 1:2 * m:2] = -q.imag       # Pr Xi
    r[:, 2 * m::2] = -q.imag        # Pi Xr
    r[:, 2 * m + 1::2] = -q.real    # Pi Xi
    return a, r


def integrate_pixels_fourier(a, r, x_half, tmp, work, xp):
    """
    g @ irfft2(x_half[b], s=(N, N), norm='forward') @ g.T for each b, from the
    half spectra x_half, with (a, r) from fourier_integration_matrices(g, N).
    This replaces the inverse FFT and integrate_pixels() when the intensity
    is available in Fourier space (e.g. after a convolution).

    Parameters
    ----------
    a, r : arrays (2 npx, N) and (npx, 4 M), M = N // 2 + 1, real
    x_half : array (nb, N, M), complex, C-contiguous
    tmp : array (nb, 2 npx, 2 M), real, contiguous
        Work buffer.
    work : array (nb, npx, npx), real, contiguous
        Output.
    xp : module
        numpy or cupy.

    Returns
    -------
    image : array (nb, npx, npx), the `work` array
    """
    nb, npx = work.shape[:2]
    left_mul(a, x_half.view(tmp.dtype), tmp, xp)
    dot_rows(tmp.reshape(nb * npx, -1), r, work.reshape(nb * npx, npx), xp)
    return work
