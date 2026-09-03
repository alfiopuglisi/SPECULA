"""
Centralized terminal output arbitration.

`TerminalInput` reads user commands from a subprocess that shares the same
terminal device as the main SPECULA process, which produces output via the
`logging` module (from many different objects) and, occasionally, via stray
`print()` calls. Without any coordination, these independent writers of the
same terminal can interleave with the input prompt, corrupting whatever the
user is currently typing.

This module centralizes all terminal writes so that:

- logging records are pushed onto a queue (`logging.handlers.QueueHandler`)
  and rendered to the terminal by a single consumer thread
  (`logging.handlers.QueueListener`);
- `sys.stdout`/`sys.stderr` are wrapped so that stray `print()` calls go
  through the same synchronized write path instead of writing directly;
- both paths serialize their actual writes using `terminal_lock`, a
  `multiprocessing.Lock` that is also shared explicitly with the
  `TerminalInput` child process. That process holds the lock for the whole
  duration of a prompt/input cycle, so no other writer can touch the
  terminal while the user is composing a command line.
"""
import sys
import queue
import logging
import logging.handlers
import multiprocessing as mp

# Shared with the TerminalInput child process. It is passed explicitly as
# an argument to mp.Process (instead of just being imported), so that it
# works correctly with both the "fork" and "spawn" multiprocessing start
# methods.
#
# This must be an RLock (not a plain Lock): the child process inherits
# sys.stdout already wrapped by _LockedStream (since it is forked/spawned
# after install() runs), so input()'s own internal write of the prompt
# re-acquires the same lock while terminal_task() is already holding it
# for the whole prompt/input cycle. A plain, non-reentrant Lock would
# self-deadlock in that case.
terminal_lock = mp.RLock()

_installed = False
_orig_stdout = None
_orig_stderr = None
_orig_handlers = None
_log_queue = None
_listener = None


class _LockedStream:
    """
    File-like wrapper that serializes writes to the real stream using
    `terminal_lock`, so that print() output cannot interleave with an
    input prompt being edited in the TerminalInput child process.
    """

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if not text:
            return 0
        with terminal_lock:
            self._stream.write(text)
            self._stream.flush()
        return len(text)

    def flush(self):
        with terminal_lock:
            self._stream.flush()

    def isatty(self):
        return getattr(self._stream, 'isatty', lambda: False)()


class _RenderHandler(logging.Handler):
    """
    Handler that actually formats and writes log records to the real
    terminal. It is only ever called from the single QueueListener
    consumer thread, but the write itself is still serialized with
    `terminal_lock` since it is shared with the TerminalInput process.
    """

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
        except Exception:
            self.handleError(record)
            return
        stream = _orig_stdout or sys.__stdout__
        with terminal_lock:
            stream.write(msg)
            stream.flush()


def install():
    """
    Install the centralized terminal output arbitration.

    Replaces the root logger handlers with a `QueueHandler` feeding a
    single consumer thread (`QueueListener`) that renders the formatted
    records, and redirects `sys.stdout`/`sys.stderr` so that stray
    `print()` calls are serialized through the same lock.

    Safe to call more than once: subsequent calls are no-ops until
    `uninstall()` is called.
    """
    global _installed, _orig_stdout, _orig_stderr, _orig_handlers
    global _log_queue, _listener

    if _installed:
        return

    root = logging.getLogger()
    _orig_handlers = list(root.handlers)

    render_handler = _RenderHandler()
    for h in _orig_handlers:
        if h.formatter is not None:
            render_handler.setFormatter(h.formatter)
        for f in h.filters:
            render_handler.addFilter(f)

    _log_queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(_log_queue)
    root.handlers = [queue_handler]

    _listener = logging.handlers.QueueListener(
        _log_queue, render_handler, respect_handler_level=True)
    _listener.start()

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _LockedStream(_orig_stdout)
    sys.stderr = _LockedStream(_orig_stderr)

    _installed = True


def uninstall():
    """
    Undo `install()`: stop the consumer thread/listener and restore the
    original `sys.stdout`/`sys.stderr` and root logger handlers.

    Safe to call even if `install()` was never called, or was already
    undone.
    """
    global _installed, _orig_stdout, _orig_stderr, _orig_handlers, _listener

    if not _installed:
        return

    if _listener is not None:
        _listener.stop()
        _listener = None

    if _orig_handlers is not None:
        logging.getLogger().handlers = _orig_handlers
        _orig_handlers = None

    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
        _orig_stdout = None
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr
        _orig_stderr = None

    _installed = False
