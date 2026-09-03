
import sys

from specula import terminal_io
from specula.processing_objects.specula_input import SpeculaInput

output_list_for_help = None

class TerminalInput(SpeculaInput):
    """
    Terminal input processing object. Handles input from a terminal.
    """

    # Override __new__ to make sure that
    # only one instance can be allocated.
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError("Only one instance of TerminalInput is allowed")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 output_list: list,
                 target_device_idx: int=None,
                 precision: int =None):
        global output_list_for_help

        """
        output_list: list of strings
            List of output names to be generated
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(output_list,
                         target_device_idx=target_device_idx,
                         precision=precision)

        output_list_for_help = output_list

        # Centralize all terminal writes (logging + stray print()) so
        # that they cannot interleave with the input prompt handled by
        # terminal_task() in the child process below. The lock is kept
        # as an attribute for introspection/testing.
        terminal_io.install()
        self.lock = terminal_io.terminal_lock
        self.set_input_task(terminal_task, self.lock)

    def finalize(self):
        super().finalize()
        terminal_io.uninstall()
        if self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=1.0)


def terminal_task(q, lock):
    sys.stdin = open(0)

    while True:
        try:
            # Hold the lock for the whole prompt/input cycle, so that no
            # other terminal writer (logging or print()) can corrupt the
            # prompt while the user is typing.
            with lock:
                tokens = [x.strip() for x in input('specula>').split()]
            if len(tokens) == 0:
                continue
            elif len(tokens) == 1:
                if tokens[0] == 'help':
                    with lock:
                        print_help()
                else:
                    q.put((tokens[0], False))
            elif len(tokens) == 2:
                value = tokens[1]
                q.put((tokens[0], value))
            else:
                with lock:
                    print('Input not recognized')
        except EOFError:
            break
        except Exception as e:
            with lock:
                print(e)

def print_help():
    print(output_list_for_help)


