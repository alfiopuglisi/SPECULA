"""
True terminal separation for interactive input, using a dedicated
console/pane on both POSIX and Windows.

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

The robust fix is to not share the terminal at all:

- On POSIX, if the simulation is running inside a tmux session, we
  open a brand new pane dedicated to the interactive prompt (own pty).
  Communication with the pane is done with a simple FIFO: the pane
  process runs a tiny read-eval-print loop that writes each completed
  line to the FIFO, and the caller reads lines back out of it.
- On Windows, tmux is not available, so instead a brand new console
  window is spawned (`subprocess.Popen(..., creationflags=
  subprocess.CREATE_NEW_CONSOLE)`) running the same kind of tiny
  read-eval-print loop. There is no FIFO equivalent on Windows, so
  communication uses a named pipe via
  `multiprocessing.connection.Listener`/`Client` instead.

In both cases the new pane/console has its own input surface, so
keystrokes and prompt redraws are physically isolated from whatever
the simulation prints in the original terminal -- no locking of any
kind is needed between the two.

If neither mechanism is usable (not on a real tty, not inside tmux on
POSIX, or unable to spawn a new console on Windows), `spawn_input_pane()`
returns None and the caller should fall back to reading input on the
current terminal.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

# Marker/prefix used to recognize the addresses returned for the
# Windows named pipe transport, as opposed to plain POSIX FIFO paths.
_WINDOWS_PIPE_MARKER = "\\\\.\\pipe\\"
_WINDOWS_PIPE_PREFIX = _WINDOWS_PIPE_MARKER + "specula_terminal_"


def tmux_available():
    """
    Return True if we are running inside a tmux session (so that
    `tmux split-window` can add a pane to it) and the `tmux` binary is
    reachable.
    """
    return 'TMUX' in os.environ and shutil.which('tmux') is not None


def _is_windows_pipe_address(address):
    return isinstance(address, str) and address.startswith(_WINDOWS_PIPE_MARKER)


def spawn_input_pane(prompt='specula> '):
    """
    Try to open a new pane/console dedicated to interactive input.

    Returns an opaque address that the caller can pass to
    `terminal_task`/`_fifo_lines` to read completed input lines back
    (a FIFO path on POSIX, a named pipe address on Windows), or None
    if a dedicated pane could not be created, in which case the caller
    should fall back to reading input on the current terminal.
    """
    if not sys.stdout.isatty():
        return None

    if sys.platform == 'win32':
        return _spawn_windows_console_pane(prompt)
    return _spawn_tmux_pane(prompt)


def _spawn_tmux_pane(prompt):
    """
    POSIX implementation: split the current tmux window and forward
    completed input lines through a FIFO.
    """
    if not tmux_available():
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


def _spawn_windows_console_pane(prompt):
    """
    Windows implementation: open a brand new console window and
    forward completed input lines through a named pipe.

    There is no FIFO-style blocking-open on Windows to naturally
    synchronize the two ends, so the spawned console retries
    connecting to the named pipe for a few seconds, giving the
    reading side (which starts listening afterwards, in the
    `terminal_task` child process) time to become ready.
    """
    create_new_console = getattr(subprocess, 'CREATE_NEW_CONSOLE', None)
    if create_new_console is None:
        return None

    address = _WINDOWS_PIPE_PREFIX + uuid.uuid4().hex

    pane_script = (
        "import time\n"
        "from multiprocessing.connection import Client\n"
        "conn = None\n"
        "for _ in range(100):\n"
        "    try:\n"
        f"        conn = Client({address!r}, family='AF_PIPE')\n"
        "        break\n"
        "    except OSError:\n"
        "        time.sleep(0.1)\n"
        "if conn is None:\n"
        "    raise SystemExit(1)\n"
        "try:\n"
        "    while True:\n"
        "        try:\n"
        f"            line = input({prompt!r})\n"
        "        except EOFError:\n"
        "            break\n"
        "        conn.send(line)\n"
        "finally:\n"
        "    conn.close()\n"
    )

    try:
        subprocess.Popen(
            [sys.executable, '-c', pane_script],
            creationflags=create_new_console,
        )
    except OSError:
        return None

    return address


def _fifo_lines(fifo_path):
    """
    Yield successive lines forwarded by the dedicated input pane,
    whether it is a POSIX FIFO or a Windows named pipe.
    """
    if _is_windows_pipe_address(fifo_path):
        yield from _windows_pipe_lines(fifo_path)
        return

    with open(fifo_path, 'r') as f:
        for line in f:
            yield line.rstrip('\n')


def _windows_pipe_lines(address):
    """
    Yield successive lines received over the named pipe identified by
    "address". Listening is set up here (in the reader, i.e. the
    `terminal_task` child process) rather than in
    `_spawn_windows_console_pane`, since the pane's `Client` retries
    connecting until this listener is ready.
    """
    from multiprocessing.connection import Listener

    with Listener(address, family='AF_PIPE') as listener:
        with listener.accept() as conn:
            while True:
                try:
                    yield conn.recv()
                except EOFError:
                    return


def cleanup_input_pane(fifo_path):
    """
    Release resources created by `spawn_input_pane()`: remove the FIFO
    (and its containing temporary directory) on POSIX, or do nothing
    on Windows (the named pipe is torn down automatically when the
    listener/connection are closed). Safe to call even if the
    resources no longer exist, or if "fifo_path" is None.
    """
    if not fifo_path or _is_windows_pipe_address(fifo_path):
        return
    try:
        os.remove(fifo_path)
    except OSError:
        pass
    try:
        os.rmdir(os.path.dirname(fifo_path))
    except OSError:
        pass
