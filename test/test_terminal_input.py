
import unittest

from specula.processing_objects.terminal_input import TerminalInput


class TestTerminalInput(unittest.TestCase):

    def setUp(self):
        TerminalInput._instance = None

    def tearDown(self):
        # Make sure the singleton and its child process do not leak
        # into other tests.
        instance = TerminalInput._instance
        if instance is not None:
            instance.finalize()
        TerminalInput._instance = None

    def test_singleton(self):
        a = TerminalInput(output_list=["a:int", "b:float"])

        with self.assertRaises(RuntimeError):
            b = TerminalInput(output_list=["a:int", "b:float"])

    def test_no_tmux_pane_without_tty_or_tmux_session(self):
        # In a non-interactive/test environment (no real tty, no TMUX
        # session) TerminalInput must fall back to reading its own
        # terminal directly, never hang or raise trying to spawn a pane.
        a = TerminalInput(output_list=["a:int", "b:float"])
        self.assertIsNone(a.fifo_path)

    def test_prompt_lines_never_blocks_on_input(self):
        # Regression test for the reported hang: input/output used to
        # share a lock that was held across the blocking input() call,
        # so a concurrent print()/log write could stall forever until
        # the user finished typing. _prompt_lines() must never involve
        # any such coordination: input() is simply called and its
        # result yielded directly.
        from unittest import mock
        from specula.processing_objects.terminal_input import _prompt_lines

        with mock.patch('builtins.input', side_effect=EOFError):
            self.assertEqual(list(_prompt_lines()), [])
