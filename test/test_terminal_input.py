
import sys
import unittest

from specula.lib import terminal_io
from specula.processing_objects.terminal_input import TerminalInput


class TestTerminalInput(unittest.TestCase):

    def setUp(self):
        TerminalInput._instance = None

    def tearDown(self):
        # Make sure the singleton, its child process and the terminal
        # output arbitration do not leak into other tests.
        instance = TerminalInput._instance
        if instance is not None:
            instance.finalize()
        TerminalInput._instance = None
        terminal_io.uninstall()

    def test_singleton(self):
        a = TerminalInput(output_list=["a:int", "b:float"])

        with self.assertRaises(RuntimeError):
            b = TerminalInput(output_list=["a:int", "b:float"])

    def test_installs_and_uninstalls_terminal_io(self):
        orig_stdout = sys.stdout

        a = TerminalInput(output_list=["a:int", "b:float"])
        self.assertIsNot(sys.stdout, orig_stdout)

        a.finalize()
        self.assertIs(sys.stdout, orig_stdout)

    def test_shares_lock_with_child_process(self):
        a = TerminalInput(output_list=["a:int", "b:float"])
        self.assertIs(terminal_io.terminal_lock, a.lock)
