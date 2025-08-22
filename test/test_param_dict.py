

import specula
from specula.param_dict import ParamDict
specula.init(0)  # Default target device

import copy
import unittest


class TestParamDict(unittest.TestCase):


    def test_delayed_input(self):
        '''This test checks that the has_delayed_input method of
        Simul returns True if any object has a delayed input with
        the -1 syntax.
        '''
        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
            'obj3': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output'
                }
            }      
        }
        params = ParamDict()
        params.params = pars
        
        assert params.has_delayed_output('obj1') == True
        assert params.has_delayed_output('obj2') == False

    def test_combine_params(self):

        original_params = {
            'dm': { 'foo' : 'bar'},
            'dm2': { 'foo2': 'bar2'},
        }
        additional_params1 = {'dm_override_2': { 'foo': 'bar3' } }
        additional_params2 = {'remove_3': ['dm2'] }

        params = ParamDict()

        # Nothing happens for simul_idx=1 (not referenced in additional_params)
        params.params = copy.deepcopy(original_params)
        params.combine_params(additional_params1, simul_idx=1)
        assert params.params == original_params

        # DM is overridden
        params.params = copy.deepcopy(original_params)
        params.combine_params(additional_params1, simul_idx=2)
        assert params.params['dm']['foo'] == 'bar3'              # Changed
        assert params.params['dm2'] == original_params['dm2']    # Unchanged

        # DM2 is removed
        params.params = copy.deepcopy(original_params)
        params.combine_params(additional_params2, simul_idx=3)
        assert params.params['dm'] == original_params['dm']      # Unchanged
        assert 'dm2' not in params.params
