from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity


class SHWrapper(BaseProcessingObj):
    """
    Shared lifecycle for processing objects that wrap multiple SH instances.
    """

    def __init__(self, ccd_side, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self._wfs_instances = []
        self._out_i = Intensity(
            ccd_side,
            ccd_side,
            precision=self.precision,
            target_device_idx=self.target_device_idx,
        )

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self._out_i

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil')}

    @classmethod
    def output_names(cls):
        return {'out_i': OutputDesc(Intensity, 'Output intensity on the detector')}

    def setup_child_inputs(self):
        pass

    def prepare_child_inputs(self):
        pass

    def reset_output(self):
        self._out_i.i[:] = 0.0

    def accumulate_output(self):
        raise NotImplementedError

    def finalize_output(self):
        pass

    def setup(self):
        super().setup()
        self.setup_child_inputs()
        for wfs in self._wfs_instances:
            wfs.setup()

    def check_ready(self, t):
        super().check_ready(t)
        for wfs in self._wfs_instances:
            wfs.check_ready(t)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.prepare_child_inputs()
        for wfs in self._wfs_instances:
            wfs.prepare_trigger(t)

    def trigger_code(self):
        self.reset_output()
        for wfs in self._wfs_instances:
            wfs.trigger_code()

    def post_trigger(self):
        super().post_trigger()
        for wfs in self._wfs_instances:
            wfs.post_trigger()
        self.accumulate_output()
        self._out_i.generation_time = self.current_time
        self.finalize_output()
