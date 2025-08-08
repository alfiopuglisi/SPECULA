

import specula
from specula.param_dict import ParamDict
specula.init(0)  # Default target device

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
