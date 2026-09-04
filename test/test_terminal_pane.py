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

    def test_spawn_input_pane_returns_none_without_tmux_session_or_display(self):
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

    # --- Windows-specific paths, exercised on Linux CI via mocking ---

    def test_is_windows_pipe_address(self):
        self.assertTrue(
            terminal_pane._is_windows_pipe_address(r'\\.\pipe\specula_terminal_abc'))
        self.assertFalse(terminal_pane._is_windows_pipe_address('/tmp/x/input.fifo'))
        self.assertFalse(terminal_pane._is_windows_pipe_address(None))

    def test_spawn_input_pane_dispatches_to_windows_on_win32(self):
        with mock.patch('sys.stdout.isatty', return_value=True):
            with mock.patch('sys.platform', 'win32'):
                with mock.patch.object(
                        terminal_pane, '_spawn_windows_console_pane',
                        return_value='sentinel') as spawn_windows:
                    result = terminal_pane.spawn_input_pane()
        spawn_windows.assert_called_once()
        self.assertEqual(result, 'sentinel')

    def test_spawn_windows_console_pane_returns_none_without_create_new_console(self):
        # Simulates running the win32-dispatched code where
        # CREATE_NEW_CONSOLE isn't defined on subprocess (e.g. because
        # this actually runs on a non-Windows platform).
        with mock.patch.object(terminal_pane.subprocess, 'CREATE_NEW_CONSOLE',
                                None, create=True):
            result = terminal_pane._spawn_windows_console_pane('specula> ')
        self.assertIsNone(result)

    def test_spawn_windows_console_pane_returns_address_on_success(self):
        with mock.patch.object(terminal_pane.subprocess, 'CREATE_NEW_CONSOLE',
                                0x00000010, create=True):
            with mock.patch.object(terminal_pane.subprocess, 'Popen') as popen:
                address = terminal_pane._spawn_windows_console_pane('specula> ')

        self.assertTrue(terminal_pane._is_windows_pipe_address(address))
        popen.assert_called_once()
        _, kwargs = popen.call_args
        self.assertEqual(kwargs.get('creationflags'), 0x00000010)

    def test_spawn_windows_console_pane_returns_none_when_popen_fails(self):
        with mock.patch.object(terminal_pane.subprocess, 'CREATE_NEW_CONSOLE',
                                0x00000010, create=True):
            with mock.patch.object(terminal_pane.subprocess, 'Popen',
                                    side_effect=OSError('boom')):
                address = terminal_pane._spawn_windows_console_pane('specula> ')
        self.assertIsNone(address)

    def test_fifo_lines_dispatches_windows_pipe_reader(self):
        address = r'\\.\pipe\specula_terminal_abc'
        with mock.patch.object(terminal_pane, '_windows_pipe_lines',
                                return_value=iter(['cmd1', 'cmd2'])) as reader:
            lines = list(terminal_pane._fifo_lines(address))
        reader.assert_called_once_with(address)
        self.assertEqual(lines, ['cmd1', 'cmd2'])

    def test_windows_pipe_lines_yields_received_lines(self):
        fake_conn = mock.MagicMock()
        fake_conn.recv.side_effect = ['foo', 'bar', EOFError]
        fake_conn.__enter__ = mock.Mock(return_value=fake_conn)
        fake_conn.__exit__ = mock.Mock(return_value=False)

        fake_listener = mock.MagicMock()
        fake_listener.accept.return_value = fake_conn
        fake_listener.__enter__ = mock.Mock(return_value=fake_listener)
        fake_listener.__exit__ = mock.Mock(return_value=False)

        with mock.patch('multiprocessing.connection.Listener',
                         return_value=fake_listener) as listener_cls:
            lines = list(terminal_pane._windows_pipe_lines(r'\\.\pipe\specula_terminal_abc'))

        self.assertEqual(lines, ['foo', 'bar'])
        listener_cls.assert_called_once_with(
            r'\\.\pipe\specula_terminal_abc', family='AF_PIPE')

    def test_cleanup_input_pane_is_noop_for_windows_pipe_address(self):
        # Should not attempt any filesystem operation for a Windows
        # pipe address, and must not raise.
        with mock.patch('os.remove') as remove, mock.patch('os.rmdir') as rmdir:
            terminal_pane.cleanup_input_pane(r'\\.\pipe\specula_terminal_abc')
        remove.assert_not_called()
        rmdir.assert_not_called()

    # --- Standalone terminal emulator fallback (POSIX, not in tmux) ---

    def test_graphical_session_available_false_without_display(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(terminal_pane._graphical_session_available())

    def test_graphical_session_available_true_with_display(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}):
            self.assertTrue(terminal_pane._graphical_session_available())

    def test_graphical_session_available_true_with_wayland_display(self):
        with mock.patch.dict(os.environ, {'WAYLAND_DISPLAY': 'wayland-0'}, clear=True):
            self.assertTrue(terminal_pane._graphical_session_available())

    def test_find_terminal_emulator_returns_none_when_none_installed(self):
        with mock.patch('shutil.which', return_value=None):
            path, run_option = terminal_pane._find_terminal_emulator()
        self.assertIsNone(path)
        self.assertIsNone(run_option)

    def test_find_terminal_emulator_returns_first_match(self):
        def fake_which(name):
            return '/usr/bin/xterm' if name == 'xterm' else None

        with mock.patch('shutil.which', side_effect=fake_which):
            path, run_option = terminal_pane._find_terminal_emulator()
        self.assertEqual(path, '/usr/bin/xterm')
        self.assertEqual(run_option, '-e')

    def test_spawn_input_pane_falls_back_to_terminal_emulator_without_tmux(self):
        with mock.patch('sys.stdout.isatty', return_value=True):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(
                        terminal_pane, '_spawn_terminal_emulator_pane',
                        return_value='sentinel') as spawn_emulator:
                    result = terminal_pane.spawn_input_pane()
        spawn_emulator.assert_called_once()
        self.assertEqual(result, 'sentinel')

    def test_spawn_terminal_emulator_pane_returns_none_without_display(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = terminal_pane._spawn_terminal_emulator_pane('specula> ')
        self.assertIsNone(result)

    def test_spawn_terminal_emulator_pane_returns_none_without_binary(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}):
            with mock.patch('shutil.which', return_value=None):
                result = terminal_pane._spawn_terminal_emulator_pane('specula> ')
        self.assertIsNone(result)

    def test_spawn_terminal_emulator_pane_returns_fifo_on_success(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}):
            with mock.patch('shutil.which',
                             side_effect=lambda n: '/usr/bin/xterm' if n == 'x-terminal-emulator' else None):
                with mock.patch.object(terminal_pane.subprocess, 'Popen') as popen:
                    fifo_path = terminal_pane._spawn_terminal_emulator_pane('specula> ')

        self.assertIsNotNone(fifo_path)
        self.assertTrue(os.path.exists(fifo_path))
        popen.assert_called_once()
        args, _ = popen.call_args
        self.assertEqual(args[0][:2], ['/usr/bin/xterm', '-e'])
        terminal_pane.cleanup_input_pane(fifo_path)

    def test_spawn_terminal_emulator_pane_returns_none_when_popen_fails(self):
        with mock.patch.dict(os.environ, {'DISPLAY': ':0'}):
            with mock.patch('shutil.which',
                             side_effect=lambda n: '/usr/bin/xterm' if n == 'x-terminal-emulator' else None):
                with mock.patch.object(terminal_pane.subprocess, 'Popen',
                                        side_effect=OSError('boom')):
                    fifo_path = terminal_pane._spawn_terminal_emulator_pane('specula> ')
        self.assertIsNone(fifo_path)
