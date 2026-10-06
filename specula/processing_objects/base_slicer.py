from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue

class BaseSlicer(BaseProcessingObj):
    """
    Base Slicer processing object.
    Extracts a subset of values from a BaseValue based on an index, list of indices, or a slice.
    """
    def __init__(self,
                 indices=None,   # int, list
                 slice_args=None,  # list for slice(start, stop, step)
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Validate that indices and slice_args are not both set
        if indices is not None and slice_args is not None:
            raise ValueError("Cannot specify both 'indices' and 'slice_args'")

        self.indices = indices
        # Use slice arguments to create a slice object
        if slice_args is not None:
            self.slice_obj = slice(*slice_args)
        else:
            self.slice_obj = None
        # ----------------------------------
        # Initialize output BaseValue
        n_elements = None
        if self.indices is not None:
            n_elements = 1 if isinstance(self.indices, int) else len(self.indices)
        elif self.slice_obj is not None:
            start = self.slice_obj.start
            stop = self.slice_obj.stop
            step = self.slice_obj.step if self.slice_obj.step is not None else 1
            if start is not None and stop is not None:
                n_elements = (stop - start + step - 1) // step
        value = self.xp.zeros(n_elements, dtype=self.dtype) \
            if n_elements is not None else self.xp.array([], dtype=self.dtype)
        # ----------------------------------
        self.out_value = BaseValue(value=value,
                                   target_device_idx=target_device_idx, precision=precision)
        # Set by setup(), where the output is allocated
        self._allocate_at_trigger = True
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out_value

    @classmethod
    def input_names(cls):
        return {'in_value': InputDesc(BaseValue, 'Input vector to slice')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'Output sliced vector')}

    def _slice(self, value):
        if self.indices is not None:
            return value[self.indices]
        elif self.slice_obj is not None:
            # Use slice object to extract the desired values
            return value[self.slice_obj]
        else:
            # No slicing, the whole value
            return value

    def setup(self):
        super().setup()
        # The output is allocated here, with the shape and dtype of the sliced input,
        # and then written in place: consumers can read it at a fixed address,
        # including those capturing a CUDA graph in their setup()
        value = self.local_inputs['in_value'].value
        if value is None:
            # Input not available yet (e.g. a delayed input, whose producer is set up later)
            self.logger.warning('Input in_value has no value at setup: '
                                'the output will be allocated at the first trigger')
            self._allocate_at_trigger = True
        else:
            self.out_value.value = self.xp.array(self._slice(value))
            self._allocate_at_trigger = False

    def trigger_code(self):
        sliced = self._slice(self.local_inputs['in_value'].value)
        if self._allocate_at_trigger:
            self.out_value.value = self.xp.array(sliced)
            self._allocate_at_trigger = False
        else:
            self.out_value.value[...] = sliced
        self.out_value.generation_time = self.current_time
