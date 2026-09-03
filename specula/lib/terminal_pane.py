"""
True terminal separation for interactive input, using tmux.

`TerminalInput` reads user commands from a subprocess that, by default,
shares a terminal with the main SPECULA process. That process produces
output via `logging` (from many different objects) and occasional
stray `print()` calls. Without any coordination, these independent
writers of the same terminal can interleave with the input prompt,
corrupting whatever the user is currently typing. Any scheme that
tries to fix this by serializing writes with a lock is fragile: if
that lock is ever held across the (unbounded) blocking read used to
collect a line of input, it stalls every other writer -- and,
transitively, the whole simulation -- for as long as the user takes
to type.

The robust fix is to not share the terminal at all: if the simulation
is running inside a tmux session, we can open a brand new pane
dedicated to the interactive prompt. That pane has its own pty, so
keystrokes and prompt redraws are physically isolated from whatever
the simulation prints in the original pane -- no locking of any kind
is needed between the two.

Communication between the new pane and the main process is done with
a simple FIFO: the pane process runs a tiny read-eval-print loop that
writes each completed line to the FIFO, and the caller reads lines
back out of it.

If tmux is not usable (not installed, not currently running inside a
tmux session, or stdout is not a real terminal), `spawn_input_pane()`
returns None and the caller should fall back to reading input on the
current terminal.
"""
import os
import shutil
import subprocess
import sys
import tempfile


def tmux_available():
    """
    Return True if we are running inside a tmux session (so that
    `tmux split-window` can add a pane to it) and the `tmux` binary is
    reachable.
    """
    return 'TMUX' in os.environ and shutil.which('tmux') is not None


def spawn_input_pane(prompt='specula> '):
    """
    Try to open a new tmux pane dedicated to interactive input.

    Returns the path of a FIFO that the new pane writes completed
    input lines to, or None if a dedicated pane could not be created,
    in which case the caller should fall back to reading input on the
    current terminal.
    """
    if not sys.stdout.isatty() or not tmux_available():
        return None

    fifo_dir = tempfile.mkdtemp(prefix='specula_terminal_')
    fifo_path = os.path.join(fifo_dir, 'input.fifo')
    os.mkfifo(fifo_path)

    # Minimal read-eval-print loop for the new pane: just prompt for
    # lines and forward them verbatim to the FIFO. All command parsing
    # (tokens, "help", error handling) still happens on the reading
    # side, in the main process.
    pane_script = (
        "import sys\n"
        f"with open({fifo_path!r}, 'w') as f:\n"
        "    while True:\n"
        "        try:\n"
        f"            line = input({prompt!r})\n"
        "        except EOFError:\n"
        "            break\n"
        "        f.write(line + chr(10))\n"
        "        f.flush()\n"
    )

    try:
        subprocess.run(
            ['tmux', 'split-window', '-d', sys.executable, '-c', pane_script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        cleanup_input_pane(fifo_path)
        return None

    return fifo_path


def cleanup_input_pane(fifo_path):
    """
    Remove the FIFO (and its containing temporary directory) created
    by `spawn_input_pane()`. Safe to call even if the FIFO no longer
    exists.
    """
    if not fifo_path:
        return
    try:
        os.remove(fifo_path)
    except OSError:
        pass
    try:
        os.rmdir(os.path.dirname(fifo_path))
    except OSError:
        pass
