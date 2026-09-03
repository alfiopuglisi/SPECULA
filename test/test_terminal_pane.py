import os
import unittest
from unittest import mock

from specula.lib import terminal_pane


class TestTerminalPane(unittest.TestCase):

    def test_tmux_available_false_without_tmux_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(terminal_pane.tmux_available())

    def test_tmux_available_false_without_tmux_binary(self):
        with mock.patch.dict(os.environ, {'TMUX': '/tmp/tmux-1000/default,1234,0'}):
            with mock.patch('shutil.which', return_value=None):
                self.assertFalse(terminal_pane.tmux_available())

    def test_tmux_available_true_when_both_present(self):
        with mock.patch.dict(os.environ, {'TMUX': '/tmp/tmux-1000/default,1234,0'}):
            with mock.patch('shutil.which', return_value='/usr/bin/tmux'):
                self.assertTrue(terminal_pane.tmux_available())

    def test_spawn_input_pane_returns_none_without_tty(self):
        with mock.patch('sys.stdout.isatty', return_value=False):
            self.assertIsNone(terminal_pane.spawn_input_pane())

    def test_spawn_input_pane_returns_none_without_tmux_session(self):
        with mock.patch('sys.stdout.isatty', return_value=True):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(terminal_pane.spawn_input_pane())

    def test_spawn_input_pane_returns_none_when_tmux_command_fails(self):
        with mock.patch('sys.stdout.isatty', return_value=True):
            with mock.patch.dict(os.environ, {'TMUX': '/tmp/tmux-1000/default,1234,0'}):
                with mock.patch('shutil.which', return_value='/usr/bin/tmux'):
                    with mock.patch('subprocess.run', side_effect=OSError('boom')):
                        fifo_path = terminal_pane.spawn_input_pane()
        self.assertIsNone(fifo_path)

    def test_cleanup_input_pane_removes_fifo_and_dir(self):
        import tempfile
        fifo_dir = tempfile.mkdtemp(prefix='specula_terminal_test_')
        fifo_path = os.path.join(fifo_dir, 'input.fifo')
        os.mkfifo(fifo_path)

        terminal_pane.cleanup_input_pane(fifo_path)

        self.assertFalse(os.path.exists(fifo_path))
        self.assertFalse(os.path.exists(fifo_dir))

    def test_cleanup_input_pane_handles_none(self):
        # Should not raise.
        terminal_pane.cleanup_input_pane(None)

    def test_cleanup_input_pane_handles_missing_fifo(self):
        # Should not raise even if the fifo was already removed.
        terminal_pane.cleanup_input_pane('/nonexistent/path/input.fifo')
