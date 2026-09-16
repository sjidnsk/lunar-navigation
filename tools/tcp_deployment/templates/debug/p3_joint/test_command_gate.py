import unittest
from command_gate import allowed
class GateTests(unittest.TestCase):
 def base(self):return dict(now=10.,path=('a',1),tracking=('a',1),state=3,command=(.2,.03),command_at=9.9,tracking_at=9.9,odom_at=9.9,vehicle_at=9.9,blocked=None)
 def test_tracking_allowed(self):self.assertTrue(allowed(**self.base()))
 def test_no_goal(self):
  x=self.base();x['path']=None;self.assertFalse(allowed(**x))
 def test_completion_failure_stops(self):
  for state in [0,5,6]:
   x=self.base();x['state']=state;self.assertFalse(allowed(**x))
 def test_old_reference_rejected(self):
  x=self.base();x['tracking']=('a',0);self.assertFalse(allowed(**x))
 def test_stale_sources_stop(self):
  for key in ['command_at','tracking_at','odom_at','vehicle_at']:
   x=self.base();x[key]=8.;self.assertFalse(allowed(**x))
 def test_manual_stop_blocks_same_goal_only(self):
  x=self.base();x['blocked']='a';self.assertFalse(allowed(**x))
  x['path']=x['tracking']=('b',1);self.assertTrue(allowed(**x))
 def test_invalid_or_excess_commands_stop(self):
  for cmd in [(float('nan'),0),(.21,0),(0,.04)]:
   x=self.base();x['command']=cmd;self.assertFalse(allowed(**x))
if __name__=='__main__':unittest.main()
