
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray

from specula.data_objects.source import Source
from specula.base_time_obj import BaseTimeObj
from specula.processing_objects.func_generator import FuncGenerator
from specula.processing_objects.atmo_evolution import AtmoEvolution
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.data_objects.layer import Layer
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestAtmoEvolution(unittest.TestCase):

    @cpu_and_gpu
    def test_atmo(self, target_device_idx, xp):
        '''Test that a basic AtmoEvolution and AtmoPropagation setup executes without exceptions'''
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.5], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        prop = AtmoPropagation(simulParams,                               
                               source_dict = {'on_axis_source': on_axis_source,
                                               'lgs1_source': lgs1_source},
                               target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)
        prop.inputs['atmo_layer_list'].set(atmo.outputs['layer_list'])

        for objlist in [[seeing, wind_speed, wind_direction], [atmo], [prop]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()
            
        ef_onaxis = cpuArray(prop.outputs['out_on_axis_source_ef'])
        ef_offaxis = cpuArray(prop.outputs['out_lgs1_source_ef'])

    @cpu_and_gpu
    def test_that_wrong_Cn2_total_is_detected(self, target_device_idx, xp):

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05)

        with self.assertRaises(ValueError):
            atmo = AtmoEvolution(simulParams,
                                L0=23,  # [m] Outer scale
                                data_dir=data_dir,
                                heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                                Cn2 = [0.2, 0.2], # Cn2 weights (total must be eq 1)
                                fov = 120.0,
                                target_device_idx=target_device_idx)

        # Total is 1, no exception raised.
        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_layer_list_type_length_and_element_types(self, target_device_idx, xp):

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05)

        atmo = AtmoEvolution(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=data_dir,
                            heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                            Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                            fov = 120.0,
                            target_device_idx=target_device_idx)
            
        assert isinstance(atmo.outputs['layer_list'], list)
        assert len(atmo.outputs['layer_list']) == 2
        
        for layer in atmo.outputs['layer_list']:
            assert isinstance(layer, Layer)

    @cpu_and_gpu
    def test_atmo_evolution_layers_are_not_reallocated(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(1)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

        id_a1 = id(atmo.outputs['layer_list'][0].field)
        id_b1 = id(atmo.outputs['layer_list'][1].field)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.check_ready(2)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

        id_a2 = id(atmo.outputs['layer_list'][0].field)
        id_b2 = id(atmo.outputs['layer_list'][1].field)

        assert id_a1 == id_a2
        assert id_b1 == id_b2

    @cpu_and_gpu
    def test_wrong_seeing_length_is_checked(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=[0.65, 0.1], target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()
            
        with self.assertRaises(ValueError):
            atmo.setup()
        
    @cpu_and_gpu
    def test_wrong_wind_speed_length_is_checked(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[8.5, 5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()
            
        with self.assertRaises(ValueError):
            atmo.setup()

    @cpu_and_gpu
    def test_wrong_wind_speed_direction_is_checked(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.2, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[90, 0, 90], target_device_idx=target_device_idx)

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for obj in [seeing, wind_speed, wind_direction]:
            obj.setup()
            
        with self.assertRaises(ValueError):
            atmo.setup()


    @cpu_and_gpu
    def test_extra_delta_time(self, target_device_idx, xp):

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=1)
    
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        seeing = FuncGenerator(constant=0.65, target_device_idx=target_device_idx)
        wind_speed = FuncGenerator(constant=[5.5, 2.3], target_device_idx=target_device_idx)
        wind_direction = FuncGenerator(constant=[0, 90], target_device_idx=target_device_idx)

        delta_time = 1.0
        delta_t = BaseTimeObj().seconds_to_t(delta_time)
        extra_delta_time = 0.1

        atmo = AtmoEvolution(simulParams,
                             L0=23,  # [m] Outer scale
                             data_dir=data_dir,
                             heights = [30.0000, 26500.0], # [m] layer heights at 0 zenith angle
                             Cn2 = [0.5, 0.5], # Cn2 weights (total must be eq 1)
                             fov = 120.0,
                             extra_delta_time=extra_delta_time,
                             target_device_idx=target_device_idx)

        atmo.inputs['seeing'].set(seeing.output)
        atmo.inputs['wind_direction'].set(wind_direction.output)
        atmo.inputs['wind_speed'].set(wind_speed.output)

        for objlist in [[seeing, wind_speed, wind_direction], [atmo]]:
            for obj in objlist:
                obj.setup()

            for obj in objlist:
                obj.check_ready(0)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

            for obj in objlist:
                obj.check_ready(delta_t)

            for obj in objlist:
                obj.trigger()

            for obj in objlist:
               obj.post_trigger()

        assert atmo.delta_time == delta_time + extra_delta_time
