import copy

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.composite_wfs import CompositeWFS
from specula.processing_objects.sh import SH
from specula import cp


class _SliceSH(SH):
    """
    SH computing a slice of the subaperture rows. The flux normalization
    is done on the whole frame by DistributedSH.
    """
    def post_trigger(self):
        BaseProcessingObj.post_trigger(self)


class DistributedSH(CompositeWFS, SH):
    """
    SH class that distributes work on multiple devices.

    Internally, it manages a series of hidden SH objects each performing
    a section of the subaperture processing. In post_trigger(), all
    results are gathered into a single Intensity array.
    Each hidden SH runs its own CUDA graph on its own device.
    """
    def __init__(self,
                 wavelengthInNm: float,
                 subap_wanted_fov: float,
                 sensor_pxscale: float,
                 subap_on_diameter: int,
                 subap_npx: int,
                 n_slices: int,
                 squaremask: bool = True,
                 fov_ovs_coeff: float = 0,
                 xShiftPhInPixel: float = 0,
                 yShiftPhInPixel: float = 0,
                 rotAnglePhInDeg: float = 0,
                 set_fov_res_to_turbpxsc: bool = False,
                 laser_launch_tel: LaserLaunchTelescope = None,
                 target_device_idx: int = None,
                 precision: int = None,
        ):
        # Complete dict of init arguments, without extra ones
        args = copy.copy(locals())
        del args['self']
        del args['__class__']

        # Calculate slices for each SH
        subaps_per_sh = subap_on_diameter // n_slices
        del args['n_slices']

        # Initialize base class - we do not use the calculation routines,
        # but it is needed for inputs and outputs (see CompositeWFS)
        super().__init__(**args)

        self.slices = []
        for i in range(n_slices):
            self.slices.append(slice( i * subaps_per_sh, (i+1) * subaps_per_sh))

        # Initialize internal SH with the other slices.
        # If using GPUs, each one targets a different device.
        self._wfs_instances = []

        for i in range(n_slices):
            if target_device_idx >= 0:
                num_devices = cp.cuda.runtime.getDeviceCount()
                args['target_device_idx'] = (target_device_idx + i) % num_devices
            args['subap_rows_slice'] = self.slices[i]
            self._wfs_instances.append(_SliceSH(**args))

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def connect_wfs_inputs(self):
        '''
        Copy our inputs into all sub-SH
        '''
        for i, sh in enumerate(self._wfs_instances):
            sh.name = f'subsh{i}'
            for k, v in self.inputs.items():
                if len(v.input_values) > 0:
                    sh.inputs[k].set(v.input_values[0].cloned_value)

    def combine_wfs_outputs(self):
        '''
        Gather results from the sub-SH and perform
        the final normalization
        '''
        for s, sh in zip(self.slices, self._wfs_instances):
            y1 = self._subap_npx * s.start
            y2 = self._subap_npx * s.stop
            self._out_i.i[y1:y2] = sh._out_i.i[y1:y2]

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i.i *= phot / self._out_i.i.sum()
        self._out_i.generation_time = self.current_time
