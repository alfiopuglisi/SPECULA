from specula.processing_objects.base_generator import BaseGenerator
from specula.data_objects.time_history import TimeHistory
from specula import SpeculaStopSimulationException


class TimeHistoryGenerator(BaseGenerator):
    """
    Generates signals from pre-computed time history data.
    """
    def __init__(self,
                 time_hist: TimeHistory,
                 stop_when_done: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):

        time_history_array = time_hist.time_history
        output_size = time_history_array.shape[1] if time_history_array.ndim > 1 else 1

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.time_hist = self.to_xp(time_history_array)
        self.stop_when_done = stop_when_done

    def trigger_code(self):
        if self.iter_counter < self.time_hist.shape[0]:
            self.output.value[:] = self.time_hist[self.iter_counter]
        else:
            # Beyond available data, use last values
            self.output.value[:] = self.time_hist[-1]

    def post_trigger(self):
        super().post_trigger()

        if self.stop_when_done and self.iter_counter == len(self.time_hist):
            raise SpeculaStopSimulationException
