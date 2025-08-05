import numpy as np
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList


class LinearCombination(BaseProcessingObj):
    def __init__(self,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)       

        self.inputs['in_vectors_list'] = InputList(type=BaseValue)
        self.out_vector = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_vector'] = self.out_vector

    def trigger_code(self):
        in_vectors = self.local_inputs['in_vectors_list']
        lgs = in_vectors[0].value
        focus = in_vectors[1].value
        lift = in_vectors[2].value
        ngs = in_vectors[3].value

        lgs[0:2] = ngs[0:2]
        lgs[2] = focus[0]
#        focus *= 0.0
#        lift *= 0
#        ngs *= 0

        self.out_vector.value *= 0.0
        self.out_vector.value[:len(lgs)] = lgs

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_vector'].set_refreshed(self.current_time)
        
    def setup(self):
        super().setup()

        in_vectors = self.local_inputs['in_vectors_list']
        lgs = in_vectors[0].value
        focus = in_vectors[1].value
        lift = in_vectors[2].value
        ngs = in_vectors[3].value

        self.out_vector.value = np.concatenate([lgs, focus, lift, ngs]) * 0.0