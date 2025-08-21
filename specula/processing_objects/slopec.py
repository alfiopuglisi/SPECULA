
from astropy.io import fits

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


def build_and_save_filtmat(intmat, recmat, nmodes, filename, xp):
    '''
    Helper functon to produce a filtering matrix,
    joining an intmat and a recmat.
    '''
    im = intmat[:nmodes, :]
    rm = recmat[:, :nmodes]
    filtmat = xp.stack((im, xp.transpose(rm)), axis=-1)
    fits.writeto(filename, cpuArray(filtmat))
    print(f'saved {filename}')


class Slopec(BaseProcessingObj):
    def __init__(self,
                 sn: Slopes=None,
                 recmat: Recmat=None,
                 filt_intmat: Intmat=None,
                 filt_recmat: Recmat=None,
                 filtmat=None,
                 weight_int_pixel_dt: float=0,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.sn = sn
        self.slopes = Slopes(self.nslopes(), target_device_idx=self.target_device_idx) 
        self.flux_per_subaperture_vector = BaseValue(self.xp.zeros(self.nsubaps(), dtype=self.dtype),
                                                     target_device_idx=self.target_device_idx)

        self.total_counts = BaseValue(self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.subap_counts = BaseValue(self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.recmat = recmat
        if filtmat is not None:
            if filt_intmat:
                raise ValueError('filt_intmat must not be set if "filtmat" is set')
            if filt_recmat:
                raise ValueError('filt_recmat must not be set if "filtmat" is set')
            self.filt_intmat = Intmat(filtmat[0], target_device_idx=self.target_device_idx)
            self.filt_recmat = Recmat(filtmat[1], target_device_idx=self.target_device_idx)
        else:
            if bool(filt_intmat) != bool(filt_recmat):
                raise ValueError('Both filt_intmat and filt_recmat must be set for slopes filtering')
            self.filt_intmat = filt_intmat
            self.filt_recmat = filt_recmat

        self.weight_int_pixel_dt = self.seconds_to_t(weight_int_pixel_dt)
        if self.weight_int_pixel_dt > 0:
            self.weight_int_pixel = True
        else:
            self.weight_int_pixel = False
        self.int_pixels = None
        self.t_previous = None

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self.slopes
        self.outputs['out_flux_per_subaperture'] = self.flux_per_subaperture_vector
        self.outputs['out_total_counts'] = self.total_counts
        self.outputs['out_subap_counts'] = self.subap_counts

    # Derived classes must implement this method
    def nsubaps(self):
        raise NotImplementedError

    # Derived classes must implement this method
    def nslopes(self):
        raise NotImplementedError

    def do_accumulation(self, t):
        """
        Perform pixel accumulation based on the IDL version.
        This method should be called in trigger_code of derived classes.
        """
        if self.weight_int_pixel_dt <= 0:
            return
 
        current_pixels = self.inputs['in_pixels'].get(self.target_device_idx).pixels.copy()

        # Calculate accumulation factor
        if self.t_previous is None:
            factor = 0.0
            delta_t = 0.0
        else:
            delta_t = t - self.t_previous
            factor = float(delta_t) / float(self.weight_int_pixel_dt)
        self.t_previous = t

        # Initialize accumulated pixels if not exists
        if self.int_pixels is None:
            from specula.data_objects.pixels import Pixels
            self.int_pixels = Pixels(
                current_pixels.shape[0],
                current_pixels.shape[1],
                target_device_idx=self.target_device_idx
            )
            self.int_pixels.pixels = self.xp.zeros_like(current_pixels)

        # Check if we're at the start of a new accumulation period
        if (t % self.weight_int_pixel_dt) == delta_t and t > self.weight_int_pixel_dt:
            # Reset accumulation
            self.int_pixels.pixels = current_pixels.astype(self.dtype) * factor
        else:
            # Add to existing accumulation
            self.int_pixels.pixels += current_pixels.astype(self.dtype) * factor

        if (t % self.weight_int_pixel_dt) == 0 and t > 0:
            # Update generation time
            self.int_pixels.generation_time = t

        if self.verbose:
            print(f'Accumulation factor is: {factor}')

    def trigger_code(self):
        raise NotImplementedError(f'{self.__class__.__name__}: please implement trigger_code() in your derived class!')

    def post_trigger(self):
        super().post_trigger()

        if self.sn:
            self.slopes.xslopes -= self.sn.xslopes
            self.slopes.yslopes -= self.sn.yslopes

        if self.recmat:
            m = self.xp.dot(self.slopes.slopes, self.recmat.recmat)
            self.slopes.slopes[:] = m

        if self.filt_intmat and self.filt_recmat:
            m = self.slopes.slopes @ self.filt_recmat.recmat
            sl0 = m @ self.filt_intmat.intmat.T
            self.slopes.slopes -= sl0

        self.outputs['out_slopes'].generation_time = self.current_time
        self.outputs['out_flux_per_subaperture'].generation_time = self.current_time
        self.outputs['out_total_counts'].generation_time = self.current_time
        self.outputs['out_subap_counts'].generation_time = self.current_time

        #rms = self.xp.sqrt(self.xp.mean(self.slopes.slopes**2))
        #print('Slopes have been filtered. '
        #      'New slopes min, max and rms: '
        #      f'{self.slopes.slopes.min()}, {self.slopes.slopes.max()}, {rms}')
