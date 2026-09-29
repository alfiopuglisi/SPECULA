import numpy as np
import scipy.ndimage
from specula import cpuArray, to_xp


class Interp2D():

    def __init__(self, input_shape, output_shape,
                 rotInDeg=0, rowShiftInPixels=0,
                 colShiftInPixels=0, magnification=1.0,
                 yy=None, xx=None, dtype=np.float32, xp=np):
        '''
        Initialize an Interp2D object for 2D interpolation between arrays.

        Parameters
        ----------
        input_shape : tuple of int
            Shape (rows, cols) of the input array to be interpolated.
        output_shape : tuple of int
            Desired shape (rows, cols) of the output (interpolated) array.
        rotInDeg : float, optional
            Rotation angle in degrees to apply to the sampling grid (default: 0).
        rowShiftInPixels : float, optional
            Vertical shift (in pixels) to apply to the sampling grid (default: 0).
        colShiftInPixels : float, optional
            Horizontal shift (in pixels) to apply to the sampling grid (default: 0).
        magnification : float, optional
            Magnification factor to apply to the sampling grid (default: 1.0).
        yy : array-like, optional
            Precomputed y-coordinates for the output grid (same shape as output_shape).
        xx : array-like, optional
            Precomputed x-coordinates for the output grid (same shape as output_shape).
        dtype : data-type, optional
            Data type for interpolation (default: np.float32).
        xp : module, optional
            Array module to use (default: numpy).

        Notes
        -----
        If `xx` and `yy` are not provided, they are generated to map the output grid
        to the input grid, with optional rotation and shift applied.

        Interpolation is bilinear, with coordinates outside the input clamped to
        the edges. It is done by scipy.ndimage on CPU and cupyx.scipy.ndimage on GPU,
        with the same calls. Whenever the sampling grid is an affine function of the
        output pixel indices (always true unless `xx` and `yy` are given, and also
        true for regular grids passed as `xx` and `yy`), it is stored as a 2x3 matrix
        and coordinates are computed on the fly. Otherwise, the coordinates are stored.
        '''
        self.xp = xp
        self.dtype = dtype
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.do_interp = True
        self.shift_x = self.dtype(colShiftInPixels)
        self.shift_y = self.dtype(rowShiftInPixels)
        self.rot_angle = rotInDeg * np.pi / 180.0
        self.magnification = self.dtype(magnification)

        # Check if interpolation is actually needed
        if (input_shape == output_shape and
            rotInDeg == 0 and
            rowShiftInPixels == 0 and
            colShiftInPixels == 0 and
            magnification == 1.0 and
            xx is None and yy is None):
            # If not, it will be skipped later
            self.do_interp = False
            return

        if xp is np:
            self.ndimage = scipy.ndimage
        else:
            import cupyx.scipy.ndimage
            self.ndimage = cupyx.scipy.ndimage

        # Base grid, before rotation, magnification and shift, as a 2x3 affine
        # matrix mapping output (row, col) to input (row, col), or as arrays.
        if xx is None or yy is None:
            base = np.array([[(input_shape[0] - 1) / output_shape[0], 0, 0],
                             [0, (input_shape[1] - 1) / output_shape[1], 0]])
        else:
            if yy.shape != output_shape or xx.shape != output_shape:
                raise ValueError(f'yy and xx must have shape {output_shape}')
            yy = cpuArray(yy, dtype=np.float64)
            xx = cpuArray(xx, dtype=np.float64)
            base = self._fit_affine(yy, xx)

        # Rotation and magnification around the input center, then shift
        cos_ = np.cos(self.rot_angle)
        sin_ = np.sin(self.rot_angle)
        lin = np.array([[cos_, sin_], [-sin_, cos_]]) / magnification
        center = np.array([input_shape[0] / 2 - 0.5, input_shape[1] / 2 - 0.5])
        shift = np.array([rowShiftInPixels, colShiftInPixels], dtype=np.float64)

        if base is not None:
            matrix = np.empty((2, 3))
            matrix[:, :2] = lin @ base[:, :2]
            matrix[:, 2] = lin @ (base[:, 2] - center) + center + shift
            # scipy computes coordinates in float64, cupyx in the input dtype
            self.matrix = matrix if xp is np else to_xp(xp, matrix, dtype=dtype)
            self.coords = None
        else:
            yc = yy - center[0]
            xc = xx - center[1]
            coords = np.stack([lin[0, 0] * yc + lin[0, 1] * xc,
                               lin[1, 0] * yc + lin[1, 1] * xc])
            coords += (center + shift)[:, None, None]
            self.matrix = None
            self.coords = to_xp(xp, coords, dtype=dtype)

        self.use_precomputed = self.coords is not None

    @staticmethod
    def _fit_affine(yy, xx):
        '''
        Return the 2x3 matrix mapping (row, col) indices to (yy, xx),
        or None if the grid is not affine.
        '''
        rows, cols = yy.shape
        if rows < 2 or cols < 2:
            return None
        r = np.arange(rows)[:, None]
        c = np.arange(cols)[None, :]
        matrix = np.empty((2, 3))
        for i, g in enumerate((yy, xx)):
            dr = (g[-1, 0] - g[0, 0]) / (rows - 1)
            dc = (g[0, -1] - g[0, 0]) / (cols - 1)
            tol = 1e-6 * max(1.0, np.abs(g).max())
            if np.abs(g - (g[0, 0] + dr * r + dc * c)).max() > tol:
                return None
            matrix[i] = dr, dc, g[0, 0]
        return matrix

    def interpolate(self, value, out=None):
        """
        Interpolates the input array to the output grid defined by the interpolator.

        Parameters
        ----------
        value : array-like
            The input array to be interpolated. Must have shape `input_shape`.
        out : array-like, optional
            Optional output array to store the result. If not provided, a new array is created.

        Returns
        -------
        out : array-like
            The interpolated array with shape `output_shape`.

        Raises
        ------
        ValueError
            If the input array does not have the expected shape.
        """
        if value.shape != self.input_shape:
            raise ValueError(f'Array to be interpolated must have shape'
                             f' {self.input_shape} instead of {value.shape}')

        # Skip interpolation if not needed
        if not self.do_interp:
            if out is None:
                return value
            else:
                out[:] = value
                return out

        if out is None:
            out = self.xp.empty(shape=self.output_shape, dtype=self.dtype)

        if self.coords is None:
            self.ndimage.affine_transform(value, self.matrix, output_shape=self.output_shape,
                                          output=out, order=1, mode='nearest')
        else:
            self.ndimage.map_coordinates(value, self.coords, output=out, order=1, mode='nearest')
        return out
