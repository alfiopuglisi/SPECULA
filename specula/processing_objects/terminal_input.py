
import sys

from specula.lib.terminal_pane import spawn_input_pane, cleanup_input_pane
from specula.lib.terminal_pane import _fifo_lines
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

        # If possible, run the interactive prompt in its own pane/console
        # (own tty on POSIX via tmux, own console window on Windows), so
        # that it is physically isolated from simulation output: input
        # and output never share the same terminal, so no coordination/
        # locking is needed between them. Otherwise fall back to sharing
        # the current terminal, exactly as before -- occasional visual
        # interleaving with output is possible in that case, but reading
        # input never blocks writers.
        self.fifo_path = spawn_input_pane()
        self.set_input_task(terminal_task, self.fifo_path)

    def finalize(self):
        super().finalize()
        if self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=1.0)
        cleanup_input_pane(self.fifo_path)


def terminal_task(q, fifo_path=None):
    """
    Read command lines (either forwarded from a dedicated input
    pane/console via "fifo_path", or directly from this process' own
    stdin), parse them and put resulting (name, value) pairs on "q".
    """
    if fifo_path:
        lines = _fifo_lines(fifo_path)
    else:
        sys.stdin = open(0)
        lines = _prompt_lines()

    for line in lines:
        tokens = [x.strip() for x in line.split()]
        if len(tokens) == 0:
            continue
        elif len(tokens) == 1:
            if tokens[0] == 'help':
                print_help()
            else:
                q.put((tokens[0], False))
        elif len(tokens) == 2:
            value = tokens[1]
            q.put((tokens[0], value))
        else:
            print('Input not recognized')


def _prompt_lines():
    """
    Yield successive input lines read directly from this process' own
    terminal.
    """
    while True:
        try:
            yield input('specula>')
        except EOFError:
            return
        except Exception as e:
            print(e)


def print_help():
    print(output_list_for_help)
