import logging
import math
from collections import namedtuple

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray, np
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pixels import Pixels
from specula.lib.interp2d import Interp2D
from specula.lib.toccd import toccd


WFS_Settings = namedtuple('WFS_Settings',
                         'sampling_ratio fft_sampling fft_padding fft_size actual_fov fft_res bin_fact')
WFS_Settings.__doc__ = '''
WFS settings

Contains the input parameters used to calculate the wfs
internal array geometries.
'''


class Lift(BaseProcessingObj):
    """
    LIFT algorithm processing object.
    Implements the LIFT algorithm for phase estimation from a focal plane image,
    as described in Meimon et al. 2010.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 nPistons: int,
                 nZern: int,
                 wavelengthInNm: float,
                 pix_scale: float,
                 cropped_size: int,
                 ifunc: IFunc,
                 ref_zern_amp,
                 npix_side: int = None,
                 ron: float = 0.0,
                 n_iter: int=20,
                 fft_res: int=2,
                 fix: bool=False,
                 target_device_idx: int = None,
                 precision: int = None):
        """

        Parameters
        ----------
        simul_params : SimulParams
            Simulation parameters object reference.
        nPistons : int [1]
            Number of piston modes in the modal base (see ifunc argument).
        nZern : int [1]
            Number of Zernike modes in the modal base (see ifunc argument).
        wavelengthInNm : float [nm]
            Wavelength in nanometers.
        pix_scale : float [arcsec/px]
            Pixel scale.
        npix_side : int [pixels]
            Number of pixels per side.
        cropped_size : int [pixels]
            Cropped size.
        ifunc : IFunc, optional
            Influence function data object.
            It must be coherent with nPistons and nZern modes, the first two zernike modes
            (if nZern>0) must be tip and tilt.
        ref_zern_amp : sequence [rad]
            Reference amplitudes for the Zernike block of the modal base, ordered exactly
            as in ifunc, i.e. starting from tip, tilt, defocus, and so on. Units are phase
            radians. It must have length nZern.
        n_iter : int [1], optional
            Number of iterations. Defaults to 20.
        fft_res : int [1], optional
            FFT resolution. Defaults to 2.
        fix : bool
            Fix flag. Defaults to False.
        target_device_idx : int [1], optional
            Target device index. Defaults to None.
        precision : int [1], optional
            Precision. Defaults to None.
        """

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        # Store parameters
        self.nPistons = int(nPistons)
        self.nZern = int(nZern)
        self.nmodes = self.nZern + self.nPistons
        self.airef = self._build_reference_coeffs(ref_zern_amp)
        self.n_iter = int(n_iter)
        self.wavelengthInNm = wavelengthInNm
        self.pix_scale = pix_scale
        self.npix_side = npix_side
        self.cropped_size = cropped_size
        self.fft_res = fft_res
        self.ron = ron
        self.fix = bool(fix)
        self.ref_tip = 0.0
        self.ref_tilt = 0.0

        self.inputs['in_pixels'] = InputValue(type=Pixels)

        self.out_pistons = BaseValue(
            value=self.xp.zeros(self.nPistons, dtype=self.dtype),
            target_device_idx=target_device_idx,
            precision=precision,
        )
        self.out_zern = BaseValue(
            value=self.xp.zeros(self.nZern, dtype=self.dtype),
            target_device_idx=target_device_idx,
            precision=precision,
        )
        self.outputs['out_pistons'] = self.out_pistons
        self.outputs['out_zern'] = self.out_zern

        self.logger.info(f"LIFT initialized with {self.nmodes} modes")

        self.ifunc = ifunc
        mask = self.ifunc.mask_inf_func

        self.settings = self.calc_geometry(
            mask.shape[0],
            self.simul_params.pixel_pitch,
            self.wavelengthInNm,
            pix_scale=self.pix_scale,
            fft_res=self.fft_res
        )

        if self.npix_side is None:
            self.npix_side = self.settings.fft_size

        self.set_modalbase(self.ifunc.influence_function,
                           mask)

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Input pixel data from the WFS detector')}

    @classmethod
    def output_names(cls):
        return {'out_pistons': OutputDesc(BaseValue, 'Estimated piston coefficients per subaperture'),
                'out_zern': OutputDesc(BaseValue, 'Estimated Zernike modal coefficients')}

    def _build_reference_coeffs(self, ref_zern_amp):
        airef = self.xp.zeros(self.nmodes, dtype=self.dtype)
        ref_zern_amp = self.to_xp(ref_zern_amp, dtype=self.dtype)
        if ref_zern_amp.ndim != 1:
            raise ValueError('ref_zern_amp must be a 1D sequence ordered from tip onward.')
        if ref_zern_amp.size > self.nZern:
            raise ValueError(
                f'ref_zern_amp has length {ref_zern_amp.size},'
                f' but is longer than nZern={self.nZern}.'
            )
        airef[self.nPistons:self.nPistons + ref_zern_amp.size] = ref_zern_amp
        return airef

    def ft_ft2(self, x):
        self.padded[self.pupil_slice, self.pupil_slice] = x
        return self.xp.fft.fftshift(self.xp.fft.fft2(self.padded, norm='ortho'))

    def computeCoG(self, frame, thFactor = 0.05):
        thImage = self.xp.maximum(frame - thFactor * self.xp.max(frame), 0)
        return self.ndimage_center_of_mass(thImage)

    def setRefTT(self, center_x, center_y, image_size):
        image_center = 0.5 * (image_size-1)
        self.ref_tip = (center_y - image_center) * self.radians_per_pixel - self.airef[0 + self.nPistons]
        self.ref_tilt = (center_x - image_center) * self.radians_per_pixel - self.airef[1 + self.nPistons]

    def calcCenter(self, frame):
        if self.fix:
            return (0.5 * frame.shape[0], 0.5 * frame.shape[1])
        yc, xc = self.computeCoG(frame)
        return (float(xc), float(yc))

    def crop(self, frame, center):
        half = self.cropped_size // 2
        row_start = math.ceil(center[1] - half)
        col_start = math.ceil(center[0] - half)
        return frame[row_start:row_start + 2 * half, col_start:col_start + 2 * half]

    @staticmethod
    def calc_geometry(phase_sampling, pixel_pitch, wavelengthInNm,
                      pix_scale, fft_res):
        """Calculate WFS geometry"""
        rad2arcsec = 206264.806247
        D = phase_sampling * pixel_pitch
        lmbda = wavelengthInNm * 1e-9
        sampling_ratio = ((lmbda / D) * rad2arcsec) / pix_scale
        ref_samp = np.max([2,np.ceil(sampling_ratio+1e-6)])
        if fft_res is not None:
            if fft_res > ref_samp: ref_samp = fft_res
        if sampling_ratio < ref_samp:
            bin_fact = ref_samp/sampling_ratio
            sampling_ratio *= bin_fact
        else: bin_fact = 1
        fft_res = sampling_ratio

        fft_sampling = round(phase_sampling/fft_res)
        fft_size = round(fft_sampling * fft_res)
        fft_res = fft_size / float(fft_sampling)
        fft_padding = fft_size - fft_sampling
        actual_fov = (lmbda / D) * (D / (D / fft_sampling)) * rad2arcsec

        return WFS_Settings(sampling_ratio, fft_sampling, fft_padding, fft_size, actual_fov, fft_res, bin_fact)

    def set_modalbase(self, modalbase, mask2d):

        self.fftSize = self.settings.fft_size
        self.gridSize = self.settings.fft_sampling
        self.radians_per_pixel = float(np.pi / (2.0 * self.settings.fft_res))
        pad = (self.fftSize - self.gridSize) // 2
        self.pupil_slice = slice(pad, pad + self.gridSize)
        self.padded = self.xp.zeros((self.fftSize, self.fftSize), dtype=self.complex_dtype)

        mask2d = cpuArray(mask2d)
        modalbase = cpuArray(modalbase)
        cpu_dtype = np.float32 if self.precision == 1 else np.float64
        resize_interp = Interp2D(
            mask2d.shape,
            (self.gridSize, self.gridSize),
            dtype=cpu_dtype,
            xp=np
        )

        valid_idx = np.nonzero(mask2d)
        f = np.zeros(mask2d.shape, dtype=cpu_dtype)
        modes = np.empty((self.nmodes, self.gridSize, self.gridSize), dtype=cpu_dtype)
        for i in range(self.nmodes):
            f[valid_idx] = modalbase[i, :]
            modes[i] = resize_interp.interpolate(f)
        self.modesCube = self.xp.asarray(modes, dtype=self.dtype)

        mask2d = resize_interp.interpolate(mask2d.astype(cpu_dtype, copy=False))
        self.mask = self.xp.array(mask2d, dtype=self.dtype)
        self._img_norm = self.dtype(1.0) / self.xp.sqrt(self.mask.sum(dtype=self.dtype))
        self._check_tip_tilt_coherence(mask2d)
        # Reference phase, including the half-pixel tilt
        self.phase_ref = self.phaseFromCoeffs(self.airef) + self._get_tlt_f(self.gridSize, self.fftSize)

        self.logger.info(f"Modal base set, gridSize={self.gridSize}, fftSize={self.fftSize}")

    def _check_tip_tilt_coherence(self, mask_cpu):
        """Verify that the modes at nPistons and nPistons+1 are linear x/y slopes (tip/tilt)."""
        if self.nmodes <= self.nPistons + 1:
            return
        n = self.gridSize
        y_ramp, x_ramp = np.indices((n, n), dtype=float)
        valid = mask_cpu > 0.5
        x_flat = x_ramp[valid]
        y_flat = y_ramp[valid]
        for label, mode_idx in (('tip', self.nPistons), ('tilt', self.nPistons + 1)):
            mode_flat = cpuArray(self.modesCube[mode_idx])[valid].astype(float)
            if mode_flat.std() == 0:
                raise ValueError(
                    f"Mode {mode_idx} (expected {label}) is flat — "
                    f"check nPistons={self.nPistons} / nZern={self.nZern} match ifunc."
                )
            r_x = float(np.corrcoef(mode_flat, x_flat)[0, 1])
            r_y = float(np.corrcoef(mode_flat, y_flat)[0, 1])
            if max(abs(r_x), abs(r_y)) < 0.9:
                raise ValueError(
                    f"Mode {mode_idx} (expected {label}) is not a linear slope "
                    f"(max|r|={max(abs(r_x), abs(r_y)):.3f}) — "
                    f"check nPistons={self.nPistons} / nZern={self.nZern} match ifunc."
                )

    def phaseFromCoeffs(self, coeffs):
        """Phase reconstruction from modal coefficients"""
        coeffs = self.to_xp(coeffs, dtype=self.dtype)
        return self.xp.tensordot(coeffs, self.modesCube, axes=1)

    @staticmethod
    def abs2(x):
        return x.real**2 + x.imag**2

    def complexField(self, phase):
        phase_lift = phase + self.phase_ref + \
            self.modesCube[0 + self.nPistons] * self.ref_tip + \
            self.modesCube[1 + self.nPistons] * self.ref_tilt
        complexField = self.mask * self.xp.exp(self.complex_dtype(1j) * phase_lift)
        return complexField, self.ft_ft2(complexField)

    def _bin_and_crop(self, img, set_total=None):
        '''Bin a FFT-sampled image to the detector sampling and crop it to npix_side'''
        new_shape = (round(img.shape[0] / self.settings.bin_fact), round(img.shape[1] / self.settings.bin_fact))
        binned = toccd(img, new_shape, set_total=set_total, xp=self.xp)
        return self.crop_or_enlarge_around_peak(binned, self.npix_side,
                                                peak_index=(new_shape[0] // 2, new_shape[1] // 2))

    def focalPlaneImageFromFFT(self, complexFieldFFT, set_flux=None):
        if set_flux is not None:
            # If a certain total flux must be set,
            # there is no need for normalization since
            # it would be overridden anyway
            img = self.abs2(complexFieldFFT)
            img *= set_flux / img.sum()
        else:
            img = self.abs2(complexFieldFFT * self._img_norm)
        return self._bin_and_crop(img)

    def calcDerivatives(self, complexField, complexFieldFFT, roi):
        '''Interaction matrix: derivative of the ROI image w.r.t. each mode'''
        conjPdTilde = self.xp.conj(complexFieldFFT) * self.complex_dtype(1j)
        columns = []
        for mode in self.modesCube:
            dI = 2.0 * (conjPdTilde * self.ft_ft2(complexField * mode)).real
            dI = self._bin_and_crop(dI, set_total=0) * self.settings.bin_fact**2
            columns.append(self.crop(dI, roi).ravel())
        return self.xp.stack(columns, axis=1)

    def computeNoiseCovarianceDiag(self, image):
        '''
        Compute noise covariance matrix.
        Only return the diagonal to avoid allocating a big matrix.
        '''
        nCov = image.ravel() + self.ron**2
        return self.xp.maximum(nCov, nCov.max() * 1e-6)

    def setPsf(self, psf):
        psf = self.xp.asarray(psf)
        yc, xc = self.computeCoG(psf)
        return self.crop_or_enlarge_around_peak(psf, int(self.npix_side),
                                                peak_index=(int(round(float(yc))), int(round(float(xc)))))

    def phaseEstimation(self, psf_orig, relTol=1e-3, absTol=1e-3):
        psf = self.setPsf(psf_orig)
        total_A_ML = self.xp.zeros(self.nmodes, dtype=self.dtype)
        currentPhaseEstimate = self.xp.zeros_like(self.mask)
        olderror = float(1e18)
        n_estimates = 1
        # Estimate initial ROI and flux
        center = self.calcCenter(psf)
        flux = self.crop(psf, center).sum()
        # Set reference TT based on initial ROI
        self.setRefTT(center[0], center[1], float(psf.shape[0]))
        for i in range(self.n_iter):
            # Store current ROI from original PSF
            psfRoi = self.crop(psf, center)
            # Compute PSF from the input phase. Keep intermediate results
            complexField, complexFieldFFT = self.complexField(currentPhaseEstimate)
            I0 = self.focalPlaneImageFromFFT(complexFieldFFT, set_flux=flux)
            # Update ROI based on new image
            newCenter = self.calcCenter(I0)
            I0roi = self.crop(I0, newCenter)
            # Renormalize flux and EF based on new ROI
            flux *= self.crop(psf, newCenter).sum() / I0roi.sum()
            norm = self.xp.sqrt(flux) / self.xp.sqrt(self.abs2(complexFieldFFT).sum())
            complexField *= norm
            complexFieldFFT *= norm
            # Calculate derivatives and the weighted least-squares reconstructor
            H = self.calcDerivatives(complexField, complexFieldFFT, newCenter)
            Rdiag = self.computeNoiseCovarianceDiag(I0roi)
            Rinv = 1 / Rdiag
            htrinv = H.T * Rinv
            P_ML = self.xp.linalg.inv(htrinv @ H) @ htrinv
            # Use reconstructor to get the delta zernike estimate
            DeltaI = (psfRoi - I0roi).ravel()
            DeltaZ = P_ML @ DeltaI
            currentPhaseEstimate += self.phaseFromCoeffs(DeltaZ)
            total_A_ML += DeltaZ
            n_estimates += 1
            # Update PSF window
            center = newCenter
            # Compute convergence criteria
            DeltaZ_cpu = cpuArray(DeltaZ)
            newerror = float((DeltaI**2 * Rinv).sum()) / (self.gridSize**2)
            improvement = (olderror - newerror) / olderror
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Iteration: {i}, estimated flux: {flux}, center_x, center_y: {center}, "
                                  f"min R, max R: {Rdiag.min()}, {Rdiag.max()}, "
                                  f"modal step: {DeltaZ_cpu}, modal estimate: {total_A_ML}, criterion: {newerror}")
            if abs(improvement) < relTol or np.mean(np.abs(DeltaZ_cpu)) < absTol:
                self.logger.debug(f'criterion reached at the {i+1} iteration')
                break

        total_A_ML[0 + self.nPistons] += self.ref_tip
        total_A_ML[1 + self.nPistons] += self.ref_tilt

        # Convert to nm and wrap to [-lambda/2, +lambda/2)
        final_coeffs = total_A_ML * self.wavelengthInNm / (2 * np.pi)
        final_coeffs = self._wrap_coefficients(final_coeffs, self.wavelengthInNm)

        return currentPhaseEstimate, final_coeffs, n_estimates

    def _wrap_coefficients(self, coeffs, wavelengthInNm):
        """
        Wrap all modal coefficients to [-wavelength/2, +wavelength/2] range.

        This ensures estimates are always within one wavelength period,
        which is critical for multi-wavelength phase unwrapping.
        """
        half_lambda = wavelengthInNm / 2.0
        return ((coeffs + half_lambda) % wavelengthInNm) - half_lambda

    def focalPlaneImageLIFT(self, phase, set_flux=None):
        '''
        Backward compatibility, called from example/test
        '''
        _, complexFieldFFT = self.complexField(self.xp.asarray(phase))
        return self.focalPlaneImageFromFFT(complexFieldFFT, set_flux=set_flux)

    def crop_or_enlarge_around_peak(self, in_array, desired_width, peak_index=None):
        '''
        Function to crop a PSF frame or enlarge it around the peak,
        depending on whether it is larger or smaller than the desired width.

        peak_index: tuple with the (r,c) integer coordinates of the
                    image center. If None, it will be found with np.argmax
        '''
        # Find the peak's location
        if peak_index is None:
            peak_index = self.xp.unravel_index(self.xp.argmax(in_array, axis=None), in_array.shape)
        # Calculate half the desired width, rounded down
        half_width = desired_width // 2
        # Calculate the top-left corner of the crop box
        start_row = max(0, peak_index[0] - half_width)
        start_col = max(0, peak_index[1] - half_width)
        # Calculate the bottom-right corner of the crop box
        end_row = min(in_array.shape[0], peak_index[0] + half_width)
        end_col = min(in_array.shape[1], peak_index[1] + half_width)
        # Crop the in_array
        cropped = in_array[start_row:end_row, start_col:end_col]
        # Check if the cropped area is smaller than desired and needs to be enlarged
        if cropped.shape[0] < desired_width or cropped.shape[1] < desired_width:
            # Calculate padding to add to each side
            padding_row = (desired_width - cropped.shape[0]) // 2
            extra_row = (desired_width - cropped.shape[0]) % 2
            padding_col = (desired_width - cropped.shape[1]) // 2
            extra_col = (desired_width - cropped.shape[1]) % 2
            # Apply padding
            cropped = self.xp.pad(cropped, ((padding_row, padding_row + extra_row),
                                    (padding_col, padding_col + extra_col)),
                            'constant', constant_values=0)
        return cropped

    def _get_tlt_f(self, pupil_size, fft_size):
        '''
        Half-pixel tilt
        '''
        xx, yy = self.xp.meshgrid(self.xp.arange(-pupil_size // 2, pupil_size // 2), self.xp.arange(-pupil_size // 2, pupil_size // 2))
        tlt_f = -2 * self.xp.pi * (xx + yy) / (2*fft_size)
        # Object precision, otherwise the whole LIFT iteration runs in complex128
        return tlt_f.astype(self.dtype)

    def trigger(self):
        psf = self.local_inputs['in_pixels'].get_value()
        _, coeffs, _ = self.phaseEstimation(psf)

        self.outputs['out_pistons'].value = coeffs[:self.nPistons]
        self.outputs['out_pistons'].generation_time = self.current_time
        self.outputs['out_zern'].value = coeffs[self.nPistons:]
        self.outputs['out_zern'].generation_time = self.current_time

        self.logger.info(f"Trigger done, coeffs={coeffs[:5]}...")

    def finalize(self):
        self.logger.info(f"Finalized")
