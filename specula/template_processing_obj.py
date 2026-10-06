from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes


class ProcessingObjName(BaseProcessingObj):
    '''Obj Name'''

    def __init__(self,
                 # object specific parameters are passed here, can have a default value or not
                 # are normally provided in the yaml config file of the simulation using this
                 parameter1: float=0,       # any basic type or data object
                 parameter2: Recmat=None,   # any basic type or data object
                 # for all processing objects:                 
                 target_device_idx=None, # thise is always here
                 precision=None          # thise is always here
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # initialization code, does something witht the parameters passed
        # build an intenal state, check some conditions
        self.interalParam = self.to_xp(parameter1 * parameter2.recmat)

        # BaseValue is just an example, could be any data object 
        self.result_data1 = BaseValue('some output for this processing object', target_device_idx=target_device_idx)
        # self.result_data2 = BaseValue('some other output for this processing oibject', target_device_idx=target_device_idx)        

        # allocate the inputs dictionary
        self.inputs['in_data1'] = InputValue(type=Slopes, optional=True) # Slopes is just an example, could be any data object 
        # self.inputs['in_data_list1'] = InputList(type=Slopes, optional=True)
        
        # allocate and initialize the outputs dictionary
        self.outputs['out_result_data1'] = self.result_data1
        # self.outputs['out_result_data2'] = self.result_data2
        
        # Pre-Allocate output data
        self.result_data1.value = self.xp.zeros(self.interalParam.recmat.shape[0], dtype=self.dtype)
        # self.result_data2.value = self.xp.zeros(self.interalParam.recmat.shape[1], dtype=self.dtype)
        

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        # here the computation that cannot go into a stream should be placed
        # for example computations explicitly done in numpy (not suitable for cupy translation)


    def trigger_code(self):
        # self.local_inputs dictionary has a one-to-on correspondence with self.inputs
        # and its content is automatically set up before trigger code is invoked
        # do the computation here
        data1 = self.local_inputs['in_data1']
        if data1 is None:                        
            d1 = self.xp.zeros_like(data1, dtype=self.dtype)
        else:
            d1 = self.interalParam @ data1

        # Could do this: 
        # self.result_data1.value = d1                
        # or the following:
        self.result_data1.value[:] = d1
        # this way the objset is not reallocated, necessary when using a stream!!!


    def post_trigger(self):
        # Always call super(): by default it marks all outputs as refreshed
        # at self.current_time. If the outputs have not been computed
        # in this trigger, set self.outputs_refreshed = False before calling it.
        # Single outputs can be marked with self.result_data1.set_refreshed(t)
        # or self.result_data1.set_not_refreshed() at any time during the trigger.
        # When a stream is used, Python code in trigger_code() runs only once
        # at capture time, so this must be done here or in prepare_trigger()
        super().post_trigger()


    def setup(self):
        super().setup()
        
        # when using stream capture should do:
        # super().build_stream()

        # check some initial conditions
        data1 = self.local_inputs['in_data1']
        if not data1:
            raise ValueError("'data1' must be given as an input")
        


