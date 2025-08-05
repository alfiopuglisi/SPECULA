import warnings

from specula import fuse
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.intensity import Intensity

from specula.data_objects.simul_params import SimulParams
from specula.lib.rebin import rebin2d


@fuse(kernel_name='clamp_generic')
def clamp_generic(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


class CCD(BaseProcessingObj):
    '''Simple CCD from intensity field'''
    def __init__(self,
                 simul_params: SimulParams,
                 size: int,           # TODO list=[80,80],
                 dt: float,           # TODO =0.001,
                 bandw: float,        # TODO =300.0,
                 binning: int=1,
                 photon_noise: bool=False,
                 readout_noise: bool=False,
                 excess_noise: bool=False,
                 darkcurrent_noise: bool=False,
                 background_noise: bool=False,
                 cic_noise: bool=False,
                 cte_noise: bool=False,
                 readout_level: float=0.0,
                 darkcurrent_level: float=0.0,
                 background_level: float=0.0,
                 cic_level: float=0,
                 cte_mat=None, # ??
                 quantum_eff: float=1.0,
                 pixelGains=None,
                 photon_seed: int=1,
                 readout_seed: int=2,
                 excess_seed: int=3,
                 excess_delta: float=1.0,
                 start_time: int=0,
                 ADU_gain: float=None,
                 ADU_bias: int=400,
                 emccd_gain: int=None,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if dt <= 0:
            raise ValueError(f'dt (integration time) is {dt} and must be greater than zero')
        if dt % simul_params.time_step != 0:
            raise ValueError(f'integration time dt={dt} must be a multiple of the basic simulation time_step={simul_params.time_step}')

        self.dt = self.seconds_to_t(dt)
        self.loop_dt = self.seconds_to_t(simul_params.time_step)
        self.start_time = self.seconds_to_t(start_time)

        # TODO: move this code inside the wfs
        # if wfs and background_level:
        #     # Compute sky background
        #     if background_level == 'auto':
        #         if background_noise:
        #             surf = (self.pixel_pupil * self.pixel_pitch) ** 2. / 4. * math.pi

        #             if sky_bg_norm:
        #                 if isinstance(wfs, ModulatedPyramid):
        #                     subaps = round(wfs.pup_diam ** 2. / 4. * math.pi)
        #                     tot_pix = subaps * 4.
        #                     fov = wfs.fov ** 2. / 4. * math.pi
        #                 elif isinstance(wfs, SH):
        #                     subaps = round(wfs.subap_on_diameter ** 2. / 4. * math.pi)
        #                     if subaps != 1 and subaps < 4.:
        #                         subaps = 4.
        #                     tot_pix = subaps * wfs.sensor_npx ** 2.
        #                     fov = wfs.sensor_fov ** 2   # This is correct because it matches tot_pix, which is square as well
        #                 else:
        #                     raise ValueError(f'Unsupported WFS class: {type(wfs)}')
        #                 background_level = \
        #                     sky_bg_norm * dt * fov * surf / tot_pix * binning ** 2
        #             else:
        #                 raise ValueError('sky_bg_norm key must be set to update background_level key')
        #         else:
        #             background_level = 0

        if cte_noise and cte_mat is None:
            raise ValueError('CTE matrix must be set if CTE noise is activated')

        self._photon_noise = photon_noise
        self._readout_noise = readout_noise
        self._darkcurrent_noise = darkcurrent_noise
        self._background_noise = background_noise
        self._cic_noise = cic_noise
        self._cte_noise = cte_noise
        self._excess_noise = excess_noise

        # Adjust ADU / EM gain values
        if self._excess_noise:
            if emccd_gain is not None:
                self._emccd_gain = float(emccd_gain)
            else:
                self._emccd_gain = 400.0
            if ADU_gain is not None:
                self._ADU_gain = float(ADU_gain)
            else:
                self._ADU_gain = 1 / 20
        else:
            if emccd_gain is not None:
                warnings.warn('ATTENTION: emccd_gain will not be used if excess_noise is False',
                    RuntimeWarning)

            self._emccd_gain = 1.0
            if ADU_gain is not None:
                self._ADU_gain = float(ADU_gain)
            else:
                self._ADU_gain = 8.0

        if self._ADU_gain <= 1 and (not excess_noise or self._emccd_gain <= 1):
            warnings.warn('ATTENTION: ADU gain is less than 1 and there is no electronic multiplication.',
                RuntimeWarning)

        self._readout_level = readout_level
        # readout noise is scaled by the emccd gain because it is applied after the EMCCD gain
        # but it is defined in photo-electrons
        if self._excess_noise:
            self._readout_level *= self._emccd_gain
        self._darkcurrent_level = darkcurrent_level
        self._background_level = background_level
        self._cic_level = cic_level

        self._binning = binning
        self._cte_mat = cte_mat if cte_mat is not None else self.xp.zeros((size[0], size[1], 2), dtype=self.dtype)
        self._qe = quantum_eff

        self._pixels = Pixels(size[0] // binning, size[1] // binning, target_device_idx=target_device_idx)
        self._integrated_i = Intensity(size[0], size[1], target_device_idx=target_device_idx, precision=precision)
        self._output_integrated_i = Intensity(size[0], size[1], target_device_idx=target_device_idx, precision=precision)
        self.refresh_outputs = False

        self._photon_seed = photon_seed
        self._readout_seed = readout_seed
        self._excess_seed = excess_seed

        self._excess_delta = excess_delta
        self._ADU_bias = ADU_bias
        self._bandw = bandw
        self._pixelGains = pixelGains
        self._photon_rng = self.xp.random.default_rng(self._photon_seed)
        self._readout_rng = self.xp.random.default_rng(self._readout_seed)
        self._excess_rng = self.xp.random.default_rng(self._excess_seed)

        self.inputs['in_i'] = InputValue(type=Intensity)
        self.outputs['out_pixels'] = self._pixels
        self.outputs['out_integrated_i'] = self._output_integrated_i

        # TODO not used yet
        self._keep_ADU_bias = False
        self._bg_remove_average = False
        self._do_not_remove_dark = False
        self._notUniformQeMatrix = None
        self._one_over_notUniformQeMatrix = None
        self._notUniformQe = False
        self._normNotUniformQe = False
        self._gaussian_noise = None

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.refresh_outputs = False

    def trigger_code(self):
        if self.start_time > 0 and self.current_time < self.start_time:
            return

        self._integrated_i.sum(self.local_inputs['in_i'],
                               factor=self.t_to_seconds(self.loop_dt) * self._bandw)

        if (self.current_time + self.loop_dt - self.dt - self.start_time) % self.dt == 0:
            self.apply_binning()
            self.apply_qe()
            self.apply_noise()

            # Copy integrated intensity into output and then reset it.
            self._output_integrated_i.i[:] = self._integrated_i.i
            self._integrated_i.i *= 0.0
            self.refresh_outputs = True

    def post_trigger(self):
        super().post_trigger()
        if self.refresh_outputs:
            self.outputs['out_pixels'].set_refreshed(self.current_time)
            self.outputs['out_integrated_i'].set_refreshed(self.current_time)
        else:
            self.outputs['out_pixels'].set_not_refreshed()
            self.outputs['out_integrated_i'].set_not_refreshed()

    def apply_noise(self):
        pixels = self._pixels.pixels  # Name change, same reference

        if self._background_noise or self._darkcurrent_noise:
            pixels += (self._background_level + self._darkcurrent_level)

        if self._cte_noise:
            pixels[:] = self.xp.dot(self.xp.dot(self._cte_mat[:, :, 0], pixels), self._cte_mat[:, :, 1])

        if self._cic_noise:
            pixels += self.xp.random.binomial(1, self._cic_level, pixels.shape)
        
        if self._photon_noise:
            pixels[:] = self._photon_rng.poisson(pixels)

        if self._excess_noise:
            ex_ccd_frame = self._excess_delta * pixels
            clamp_generic(1e-10, 1e-10, ex_ccd_frame, xp=self.xp)
            pixels[:] = 1.0 / self._excess_delta * self._excess_rng.gamma(shape=ex_ccd_frame, scale=self._emccd_gain)

        if self._readout_noise:
            ron_vector = self._readout_rng.standard_normal(size=pixels.size)
            pixels += (ron_vector.reshape(pixels.shape) * self._readout_level)

        if self._pixelGains is not None:
            pixels *= self._pixelGains

        if self._photon_noise:
            pixels[:] = self.xp.round(pixels * self._ADU_gain) + self._ADU_bias
            clamp_generic(0, 0, pixels, xp=self.xp)

            if not self._keep_ADU_bias:
                pixels -= self._ADU_bias

            pixels[:] = (pixels / self._ADU_gain)
            if self._excess_noise:
                pixels[:] = (pixels / self._emccd_gain)
            if self._darkcurrent_noise and not self._do_not_remove_dark:
                pixels -= self._darkcurrent_level

            # TODO not used yet
            if self._bg_remove_average and not self._do_not_remove_dark:
                pixels -= self._background_level

        # TODO not used yet
        if self._notUniformQe and self._normNotUniformQe:
            if self._one_over_notUniformQeMatrix is None:
                self._one_over_notUniformQeMatrix = 1 / self._notUniformQeMatrix
            pixels *= self._one_over_notUniformQeMatrix


    def apply_binning(self):
        intensity = self._integrated_i.i

        if self._binning > 1:
            newshape = (intensity.shape[0] // self._binning, intensity.shape[1] // self._binning)
            self._pixels.pixels[:] = rebin2d(intensity, newshape, xp=self.xp)
        else:
            self._pixels.pixels[:] = intensity

    def apply_qe(self):
        if self._qe != 1:
            self._pixels.multiply(self._qe)
        if self._notUniformQe:
            self._pixels.multiply(self._notUniformQeMatrix)

    def setup(self):
        super().setup()
