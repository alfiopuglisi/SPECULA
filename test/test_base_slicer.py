import specula
specula.init(0)

from specula.loop_control import LoopControl

import unittest
from specula import cpuArray, np
from specula.processing_objects.base_slicer import BaseSlicer
from specula.base_value import BaseValue

class TestBaseSlicer(unittest.TestCase):

    def test_indices(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(indices=[1, 3, 5])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [1, 3, 5])

    def test_output_allocated_in_setup_and_written_in_place(self):
        value = BaseValue(value=np.arange(10))
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(indices=[1, 3, 5])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.start(run_time=2, dt=1, t0=1)
        # Allocated in setup, before the first trigger
        out = slicer.outputs['out_value'].value
        assert out.shape == (3,)
        loop.iter()
        loop.iter()

        assert slicer.outputs['out_value'].value is out
        np.testing.assert_array_equal(cpuArray(out), [1, 3, 5])

    def test_open_slice_shape_and_dtype_from_input(self):
        # Open-ended slice and input with a different precision: the output
        # size and dtype are not known when the slicer is created
        value = BaseValue(value=np.arange(10), precision=0)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[4, None], precision=1)
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.start(run_time=1, dt=1, t0=1)
        out = slicer.outputs['out_value'].value
        assert out.shape == (6,)
        assert out.dtype == value.value.dtype
        assert out.dtype != slicer.dtype
        loop.iter()

        assert slicer.outputs['out_value'].value is out
        np.testing.assert_array_equal(cpuArray(out), [4, 5, 6, 7, 8, 9])

    def test_input_without_value_at_setup(self):
        # E.g. a delayed input whose producer is set up later:
        # the output is allocated at the first trigger
        value = BaseValue()
        slicer = BaseSlicer(indices=[0, 2])
        slicer.inputs['in_value'].set(value)
        slicer.setup()

        value.value = np.arange(5)
        value.generation_time = 1
        slicer.check_ready(1)
        slicer.trigger()
        slicer.post_trigger()
        np.testing.assert_array_equal(cpuArray(slicer.outputs['out_value'].value), [0, 2])

    def test_slice_args(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[2, 7, 2])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [2, 4, 6])

    def test_no_args(self):
        arr = np.arange(5)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer()
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, arr)

    def test_slice_single_element(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[5, 6, 1])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [5])

    def test_slice_empty(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[3, 3, 1])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [])

    def test_slice_step_larger_than_range(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[2, 5, 10])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [2])
