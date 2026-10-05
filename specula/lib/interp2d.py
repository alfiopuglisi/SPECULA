import numpy as np
from specula import cp, to_xp
from scipy.interpolate import RegularGridInterpolator

class Interp2D():

    if cp: # pragma: no cover
        # Definition of bilinear interpolation device function
        bilinear_interp_device = r'''
            __device__ TYPE bilinear_interp(TYPE *g_in, int in_dx, int in_dy, TYPE xcoord, TYPE ycoord) {
                int xin = floor(xcoord);
                int yin = floor(ycoord);
                // Clamp neighbours at the last row/column: they get zero weight there,
                // but reading past the edge would wrap to the next row or out of bounds.
                int xin2 = min(xin + 1, in_dx - 1);
                int yin2 = min(yin + 1, in_dy - 1);

                TYPE xdist = xcoord - xin;
                TYPE ydist = ycoord - yin;

                int idx_a = yin * in_dx + xin;
                int idx_b = yin * in_dx + xin2;
                int idx_c = yin2 * in_dx + xin;
                int idx_d = yin2 * in_dx + xin2;

                return g_in[idx_a] * (1 - xdist) * (1 - ydist) +
                       g_in[idx_b] * xdist * (1 - ydist) +
                       g_in[idx_c] * ydist * (1 - xdist) +
                       g_in[idx_d] * xdist * ydist;
            }
            '''

        interp2_kernel_onthefly = bilinear_interp_device + r'''
            extern "C" __global__
            void interp2_kernel_onthefly_TYPE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, int in_dy,
                                            TYPE scale_x, TYPE scale_y,
                                            TYPE centered_offset_x, TYPE centered_offset_y,
                                            TYPE shift_x, TYPE shift_y,
                                            TYPE cos_angle, TYPE sin_angle, TYPE center_x, TYPE center_y, 
                                            TYPE magnification) {
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int x = blockIdx.x * blockDim.x + threadIdx.x;

                if ((y < out_dy) && (x < out_dx)) {
                    // Compute coordinates on-the-fly, centered for rotation and magnification
                    // (centered_offset = grid offset - center, computed in double on the host)
                    TYPE xx_centered = x * scale_x + centered_offset_x;
                    TYPE yy_centered = y * scale_y + centered_offset_y;
                    
                    // Apply magnification
                    if (magnification != 1.0) {
                        xx_centered /= magnification;
                        yy_centered /= magnification;
                    }
                    
                    // Apply rotation if necessary
                    if (cos_angle != 1.0 || sin_angle != 0.0) {
                        TYPE xcoord_rot = xx_centered * cos_angle - yy_centered * sin_angle;
                        TYPE ycoord_rot = xx_centered * sin_angle + yy_centered * cos_angle;
                        xx_centered = xcoord_rot;
                        yy_centered = ycoord_rot;
                    }
                    
                    // Restore center
                    TYPE xcoord = xx_centered + center_x;
                    TYPE ycoord = yy_centered + center_y;
                    
                    // Apply shift
                    xcoord += shift_x;
                    ycoord += shift_y;
                    
                    // Clamp to limits
                    if (xcoord < 0) xcoord = 0;
                    if (ycoord < 0) ycoord = 0;
                    if (xcoord > in_dx - 1) xcoord = in_dx - 1;
                    if (ycoord > in_dy - 1) ycoord = in_dy - 1;
                    
                    // Call bilinear interpolation
                    g_out[y * out_dx + x] = bilinear_interp(g_in, in_dx, in_dy, xcoord, ycoord);
                }
            }
            '''
        interp2_kernel_onthefly_float = \
            cp.RawKernel(interp2_kernel_onthefly.replace('TYPE', 'float'),
                         name='interp2_kernel_onthefly_float')
        interp2_kernel_onthefly_double = \
            cp.RawKernel(interp2_kernel_onthefly.replace('TYPE', 'double'),
                         name='interp2_kernel_onthefly_double')

    def __init__(self, input_shape, output_shape,
                 rotInDeg=0, rowShiftInPixels=0,
                 colShiftInPixels=0, magnification=1.0,
                 grid_scale=None, grid_offset=None,
                 dtype=np.float32, xp=np):
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
        grid_scale : tuple of float, optional
            (row, col) input pixels per output pixel of the regular output grid
            (default: (input_shape - 1) / output_shape).
        grid_offset : tuple of float, optional
            (row, col) input coordinates of the first output pixel of the regular
            output grid (default: (0, 0)).
        dtype : data-type, optional
            Data type for interpolation (default: np.float32).
        xp : module, optional
            Array module to use (default: numpy).

        Notes
        -----
        Output pixel (row, col) is mapped to input coordinates
        grid_offset + (row, col) * grid_scale, before rotation and magnification
        around the input center and shift. Coordinates are clamped to the input array.
        On GPU these coordinates are computed on the fly, without storing them;
        on CPU they are precomputed.
        '''
        self.xp = xp
        self.dtype = dtype
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.do_interp = True

        # Check if interpolation is actually needed
        if (input_shape == output_shape and
            rotInDeg == 0 and
            rowShiftInPixels == 0 and
            colShiftInPixels == 0 and
            magnification == 1.0 and
            grid_scale is None and grid_offset is None):
            # If not, it will be skipped later
            self.do_interp = False
            self.shift_x = 0.0
            self.shift_y = 0.0
            self.rot_angle = 0.0
            self.magnification = 1.0
            return

        if grid_scale is None:
            grid_scale = ((input_shape[0] - 1) / output_shape[0],
                          (input_shape[1] - 1) / output_shape[1])
        if grid_offset is None:
            grid_offset = (0, 0)

        if self.xp is cp:
            self.scale_x = self.dtype(grid_scale[1])
            self.scale_y = self.dtype(grid_scale[0])
            self.centered_offset_x = self.dtype(grid_offset[1] - (input_shape[1] / 2 - 0.5))
            self.centered_offset_y = self.dtype(grid_offset[0] - (input_shape[0] / 2 - 0.5))
            self.xx = None
            self.yy = None
        else:
            yy, xx = map(self.dtype, np.mgrid[0:output_shape[0], 0:output_shape[1]])
            # The -1 of the default grid_scale appears to be correct by comparing with IDL code
            yy = yy * grid_scale[0] + grid_offset[0]
            xx = xx * grid_scale[1] + grid_offset[1]

            if rotInDeg != 0 or magnification != 1.0:
                yc = input_shape[0] / 2 - 0.5
                xc = input_shape[1] / 2 - 0.5

                xx_centered = (xx - xc) / magnification
                yy_centered = (yy - yc) / magnification

                if rotInDeg != 0:
                    cos_ = np.cos(rotInDeg * np.pi / 180.0)
                    sin_ = np.sin(rotInDeg * np.pi / 180.0)
                    xxr = xx_centered * cos_ - yy_centered * sin_
                    yyr = xx_centered * sin_ + yy_centered * cos_
                    xx_centered = xxr
                    yy_centered = yyr

                xx = xx_centered + xc
                yy = yy_centered + yc

            if rowShiftInPixels != 0 or colShiftInPixels != 0:
                yy += rowShiftInPixels
                xx += colShiftInPixels

            yy[np.where(yy < 0)] = 0
            xx[np.where(xx < 0)] = 0
            yy[np.where(yy > input_shape[0] - 1)] = input_shape[0] - 1
            xx[np.where(xx > input_shape[1] - 1)] = input_shape[1] - 1

            self.yy = to_xp(self.xp, yy, dtype=dtype).ravel()
            self.xx = to_xp(self.xp, xx, dtype=dtype).ravel()

            self.scale_x = None
            self.scale_y = None
            self.centered_offset_x = None
            self.centered_offset_y = None

        self.shift_x = self.dtype(colShiftInPixels)
        self.shift_y = self.dtype(rowShiftInPixels)
        self.rot_angle = rotInDeg * np.pi / 180.0
        self.magnification = self.dtype(magnification)
        self.cos_angle = self.dtype(np.cos(self.rot_angle))
        self.sin_angle = self.dtype(np.sin(self.rot_angle))
        self.center_x = self.dtype(input_shape[1] / 2 - 0.5)
        self.center_y = self.dtype(input_shape[0] / 2 - 0.5)

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

        Notes
        -----
        For CPU arrays, uses scipy's RegularGridInterpolator.
        For GPU arrays (cupy), uses a custom CUDA kernel.
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

        if self.xp == cp: # pragma: no cover
            block = (16, 16)
            # Calculate grid size for non-square arrays correctly
            grid_x = (self.output_shape[1] + block[0] - 1) // block[0]
            grid_y = (self.output_shape[0] + block[1] - 1) // block[1]
            grid = (grid_x, grid_y)

            if self.dtype == cp.float32:
                kernel = self.interp2_kernel_onthefly_float
            elif self.dtype == cp.float64:
                kernel = self.interp2_kernel_onthefly_double
            else:
                raise ValueError(f'Unsupported dtype {self.dtype}')
            kernel(grid, block, (
                value, out,
                self.output_shape[1], self.output_shape[0],
                self.input_shape[1], self.input_shape[0],
                self.scale_x, self.scale_y,
                self.centered_offset_x, self.centered_offset_y,
                self.shift_x, self.shift_y,
                self.cos_angle, self.sin_angle,
                self.center_x, self.center_y,
                self.magnification))

            return out

        else:
            points = (self.xp.arange( self.input_shape[0], dtype=self.dtype),
                      self.xp.arange( self.input_shape[1], dtype=self.dtype))
            interp = RegularGridInterpolator(points,value, method='linear')
            out[:] = interp((self.yy, self.xx)).reshape(self.output_shape)
            return out
