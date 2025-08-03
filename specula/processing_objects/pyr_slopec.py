from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.base_value import BaseValue
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)

@fuse(kernel_name='clamp_generic_less1')
def clamp_generic_less1(x, c, y, xp):
    y = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)


@fuse(kernel_name='clamp_generic_more1')
def clamp_generic_more1(x, c, y, xp):
    y = xp.where(y > x, c, y)


class PyrSlopec(Slopec):
    def __init__(self,
                 pupdata: PupData,
                 sn: Slopes=None,
                 shlike: bool=False,
                 norm_factor: float=None,   # TODO =1.0,
                 thr_value: float=0,
                 slopes_from_intensity: bool=False,
                 target_device_idx: int=None,
                 precision: int=None,
                **kwargs): # is this needed??
        super().__init__(sn=sn, target_device_idx=target_device_idx, precision=precision, **kwargs)

        if shlike and slopes_from_intensity:
            raise ValueError('Both SHLIKE and SLOPES_FROM_INTENSITY parameters are set. Only one of these should be used.')

        if shlike and self.norm_factor != 0:
            raise ValueError('Both SHLIKE and NORM_FACTOR parameters are set. Only one of these should be used.')

        self.shlike = shlike
        self.norm_factor = norm_factor
        self.threshold = thr_value
        self.slopes_from_intensity = slopes_from_intensity
        self.pupdata = pupdata  # Property set
        ind_pup = self.pupdata.ind_pup
        self.pup_idx  = ind_pup.flatten().astype(self.xp.int64)[ind_pup.flatten() >= 0] # Exclude -1 padding
        self.pup_idx0 = ind_pup[:, 0][ind_pup[:, 0] >= 0]  # Exclude -1 padding
        self.pup_idx1 = ind_pup[:, 1][ind_pup[:, 1] >= 0]  # Exclude -1 padding
        self.pup_idx2 = ind_pup[:, 2][ind_pup[:, 2] >= 0]  # Exclude -1 padding
        self.pup_idx3 = ind_pup[:, 3][ind_pup[:, 3] >= 0]  # Exclude -1 padding
        self.n_pup = self.pupdata.ind_pup.shape[1]
        self.n_subap = self.pupdata.ind_pup.shape[0]

        self.total_counts = BaseValue(target_device_idx=self.target_device_idx)
        self.subap_counts = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_pupdata'] = self.pupdata
        self.outputs['total_counts'] = self.total_counts
        self.outputs['subap_counts'] = self.subap_counts

    @property
    def pupdata(self):
        return self._pupdata

    @pupdata.setter
    def pupdata(self, p):
        if p is not None:
            self._pupdata = p
            # TODO replace this resize with an earlier initialization
            if self.slopes_from_intensity:
                self.slopes.resize(len(self.pupdata.ind_pup) * 4)
            else:
                self.slopes.resize(len(self.pupdata.ind_pup) * 2)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.local_inputs['in_pixels'].pixels.flatten()

    def trigger_code(self):
        # outpus:
        # total_counts : computed here
        # subap_counts : computed in post_trigger
        # slopes : computed here

#        if not self.pupdata:
#            return
#        if self.verbose:
#            print('Average pixel counts:', self.xp.sum(pixels) / len(self.pupdata.ind_pup))

        self.total_counts.value = self.xp.sum(self.flat_pixels[self.pup_idx])

        self.flat_pixels -= self.threshold

        clamp_generic_less(0,0,self.flat_pixels, xp=self.xp)
        A = self.flat_pixels[self.pup_idx0].astype(self.xp.float32)
        B = self.flat_pixels[self.pup_idx1].astype(self.xp.float32)
        C = self.flat_pixels[self.pup_idx2].astype(self.xp.float32)
        D = self.flat_pixels[self.pup_idx3].astype(self.xp.float32)

        # Compute total intensity
        self.total_intensity = self.xp.sum(self.flat_pixels[self.pup_idx])

        # Use 1-length array to allow clamp() on both GPU arrays and CPU scalars
        inv_factor = self.xp.zeros(1, dtype=self.dtype)

        if self.slopes_from_intensity:
            inv_factor[0] = self.total_intensity / (4 * self.n_subap)
            factor = 1.0 / inv_factor[0]
            self.sx = factor * self.xp.concatenate([A, B])
            self.sy = factor * self.xp.concatenate([C, D])
        else:
            if self.norm_factor is not None:
                inv_factor[0] = self.norm_factor
                factor = 1.0 / inv_factor[0]
            elif not self.shlike:
                inv_factor[0] = self.total_intensity /  self.n_subap
                factor = 1.0 / inv_factor
            else:
                inv_factor[0] = self.xp.sum(self.flat_pixels[self.pup_idx])
                factor = 1.0 / inv_factor[0]

            self.sx = (A+B-C-D) * factor
            self.sy = (B+C-A-D) * factor

        clamp_generic_more(0, 1, inv_factor, xp=self.xp)
        self.sx *= inv_factor[0]
        self.sy *= inv_factor[0]

        self.slopes.xslopes = self.sx
        self.slopes.yslopes = self.sy


    def post_trigger(self):
        super().post_trigger()

        self.subap_counts.value = self.total_counts.value / self.pupdata.n_subap
        self.outputs['out_pupdata'].set_refreshed(self.current_time)
        self.outputs['total_counts'].set_refreshed(self.current_time)
        self.outputs['subap_counts'].set_refreshed(self.current_time)

        self.slopes.single_mask = self.pupdata.single_mask()
        self.slopes.display_map = self.pupdata.display_map

