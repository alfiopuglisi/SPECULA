import io
import logging
import logging.handlers
import unittest

from specula import terminal_io


class TestTerminalIO(unittest.TestCase):

    def tearDown(self):
        terminal_io.uninstall()

    def test_install_is_idempotent(self):
        terminal_io.install()
        stdout_after_first = terminal_io._orig_stdout
        terminal_io.install()
        self.assertIs(terminal_io._orig_stdout, stdout_after_first)

    def test_uninstall_without_install_is_noop(self):
        # Should not raise even if install() was never called.
        terminal_io.uninstall()

    def test_install_redirects_stdout_and_stderr(self):
        import sys
        orig_stdout, orig_stderr = sys.stdout, sys.stderr

        terminal_io.install()
        self.assertIsNot(sys.stdout, orig_stdout)
        self.assertIsNot(sys.stderr, orig_stderr)

        terminal_io.uninstall()
        self.assertIs(sys.stdout, orig_stdout)
        self.assertIs(sys.stderr, orig_stderr)

    def test_install_replaces_root_handlers_and_uninstall_restores_them(self):
        root = logging.getLogger()
        dummy_handler = logging.StreamHandler(io.StringIO())
        root.addHandler(dummy_handler)
        orig_handlers = list(root.handlers)

        try:
            terminal_io.install()
            self.assertEqual(len(root.handlers), 1)
            self.assertIsInstance(root.handlers[0], logging.handlers.QueueHandler)

            terminal_io.uninstall()
            self.assertEqual(root.handlers, orig_handlers)
        finally:
            root.removeHandler(dummy_handler)

    def test_locked_stream_writes_through_to_underlying_stream(self):
        buf = io.StringIO()
        stream = terminal_io._LockedStream(buf)
        stream.write('hello')
        self.assertEqual(buf.getvalue(), 'hello')

    def test_print_is_routed_through_terminal_lock(self):
        import builtins
        import sys
        import threading

        terminal_io.install()
        try:
            self.assertIsInstance(sys.stdout, terminal_io._LockedStream)
            # The lock is an RLock (reentrant within the same process/
            # thread, e.g. for terminal_task()'s own input() prompt
            # write), but it must still block a *different* thread.
            terminal_io.terminal_lock.acquire()
            try:
                other_thread_acquired = []

                def _try_acquire():
                    other_thread_acquired.append(
                        terminal_io.terminal_lock.acquire(block=False))

                t = threading.Thread(target=_try_acquire)
                t.start()
                t.join(timeout=2.0)
                self.assertEqual(other_thread_acquired, [False])
            finally:
                terminal_io.terminal_lock.release()
        finally:
            terminal_io.uninstall()
