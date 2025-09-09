from specula.data_objects.time_history import TimeHistory
from specula.processing_objects.time_history_generator import TimeHistoryGenerator
from specula.lib.modal_pushpull_signal import modal_pushpull_signal


class PushPullGenerator(TimeHistoryGenerator):
    """
    Generates push-pull signals for modal calibration.
    
    Extracted from BaseGenerator's PUSH and PUSHPULL types.
    """
    def __init__(self,
                 nmodes: int,
                 first_mode: int=0,
                 push_pull_type: str = 'PUSHPULL',  # 'PUSH' or 'PUSHPULL'
                 amp: float = None,
                 constant_amp: bool=False,
                 pattern: list = [1, -1],
                 vect_amplitude: list = None,
                 ncycles: int = 1,
                 nsamples: int = 1,
                 repeat_cycles: bool = False,
                 stop_when_done: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):

        if amp is None and vect_amplitude is None:
            raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH/PUSHPULL')

        if nsamples != 1 and push_pull_type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL type')

        self.stop_when_done = stop_when_done

        self.push_pull_type = push_pull_type.upper()

        # Generate the time history using modal_pushpull_signal (from original)
        if self.push_pull_type == 'PUSH':
            time_hist = modal_pushpull_signal(
                nmodes,
                first_mode=first_mode,
                amplitude=amp,
                constant=constant_amp,
                vect_amplitude=vect_amplitude,
                only_push=True,
                ncycles=ncycles
            )
        elif self.push_pull_type == 'PUSHPULL':
            time_hist = modal_pushpull_signal(
                nmodes,
                first_mode=first_mode,
                amplitude=amp,
                constant=constant_amp,
                vect_amplitude=vect_amplitude,
                pattern=pattern,
                ncycles=ncycles,
                repeat_ncycles=repeat_cycles,
                nsamples=nsamples
            )
        else:
            raise ValueError(f'Unknown push_pull_type: {self.push_pull_type}')

        super().__init__(
            time_hist=TimeHistory(time_hist),
            stop_when_done=stop_when_done,
            target_device_idx=target_device_idx,
            precision=precision,
        )
