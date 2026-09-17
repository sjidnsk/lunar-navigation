"""Exercise the real GUI callbacks without requiring ROS or a desktop."""
import ast
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'lunar_car_gui_node.py'

class Twist:
    def __init__(self):
        self.linear = SimpleNamespace(x=0.)
        self.angular = SimpleNamespace(z=0.)

class OwnershipTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LunarCarGuiNode')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('publish_cmd_vel', 'on_external_control', 'emergency_stop')]
        self.clock = SimpleNamespace(now=10.)
        ns = {'time': SimpleNamespace(monotonic=lambda: self.clock.now), 'math': math, 'Twist': Twist}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), 'exec'), ns)
        self.sent, self.modes = [], []
        self.node = SimpleNamespace(_lock=threading.Lock(), external_control_until=0., external_control_active=False,
            linear_cmd=.2, curvature_cmd=.4, publish_enabled=True,
            cmd_vel_pub=SimpleNamespace(publish=self.sent.append), send_mode=self.modes.append)
        self.methods = ns

    def call(self, name, *args):
        self.methods[name](self.node, *args)

    def test_missing_heartbeat_never_reclaims_control_or_cancels_navigation(self):
        self.call('on_external_control', SimpleNamespace(data=True))
        for age in (2., 60., 3600.):
            self.clock.now = 10. + age
            self.call('publish_cmd_vel')
        self.assertEqual(self.modes, [])
        self.assertEqual(self.sent, [])
        self.assertTrue(self.node.external_control_active)

    def test_explicit_release_parks_and_clears_old_speed_once(self):
        self.call('on_external_control', SimpleNamespace(data=True))
        self.call('on_external_control', SimpleNamespace(data=False))
        self.call('publish_cmd_vel')
        self.call('publish_cmd_vel')
        self.assertEqual(self.modes, ['park'])
        self.assertTrue(all(m.linear.x == 0. and m.angular.z == 0. for m in self.sent))

    def test_manual_emergency_stop_still_works_while_owned(self):
        self.call('on_external_control', SimpleNamespace(data=True))
        self.call('emergency_stop')
        self.assertEqual(self.modes, ['park'])
        self.assertEqual(self.sent[-1].linear.x, 0.)

    def test_manual_control_before_p4_acquires(self):
        self.call('publish_cmd_vel')
        self.assertEqual(self.sent[-1].linear.x, .2)

if __name__ == '__main__':
    unittest.main()
