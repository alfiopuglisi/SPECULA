import copy

from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_wrapper import SHWrapper
from specula import cp


class DistributedSH(SHWrapper):
    """
    SH class that distributes work on multiple devices.

    Internally, it manages a series of hidden SH objects each performing
    a section of the subaperture processing. In post_trigger(), all
    results are gathered into a single Intensity array.
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

        ccd_side = subap_on_diameter * subap_npx
        super().__init__(
            ccd_side=ccd_side,
            target_device_idx=target_device_idx,
            precision=precision,
        )
        self._subap_npx = subap_npx

        if laser_launch_tel is not None:
            self.inputs['sodium_altitude'] = InputValue(type=BaseValue, optional=True)
            self.inputs['sodium_intensity'] = InputValue(type=BaseValue, optional=True)

        self.slices = []
        for i in range(n_slices):
            self.slices.append(slice( i * subaps_per_sh, (i+1) * subaps_per_sh))

        # Initialize internal SH with the other slices.
        # If using GPUs, each one targets a different device.
        self.sub_sh = []

        for i in range(n_slices):
            if target_device_idx >= 0:
                num_devices = cp.cuda.runtime.getDeviceCount()
                args['target_device_idx'] = (target_device_idx + i) % num_devices
            args['subap_rows_slice'] = self.slices[i]
            self.sub_sh.append(SH(**args))
        self._wfs_instances = self.sub_sh

    @classmethod
    def input_names(cls):
        return SH.input_names()

    @classmethod
    def output_names(cls):
        return SH.output_names()

    def setup_child_inputs(self):
        # Copy our inputs into all sub-SH
        for i, sh in enumerate(self.sub_sh):
            sh.name = f'subsh{i}'
            for k, v in self.inputs.items():
                if len(v.input_values) > 0:
                    sh.inputs[k].set(v.input_values[0].cloned_value)

    def trigger(self):
        super(SHWrapper, self).trigger()
        for sh in self.sub_sh:
            sh.trigger()

    def accumulate_output(self):
        # Collect results from the other SHs into our Intensity result
        for s, sh in zip(self.slices, self.sub_sh):
            y1 = self._subap_npx * s.start
            y2 = self._subap_npx * s.stop
            self._out_i.i[y1:y2] = sh._out_i.i[y1:y2]

    def finalize_output(self):
        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()
        self._out_i.i *= phot / self._out_i.i.sum()
