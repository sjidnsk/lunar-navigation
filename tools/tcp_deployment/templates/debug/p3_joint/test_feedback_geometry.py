import math,unittest
from feedback_geometry import Alignment, converted_pose, PoseRates
from lunar_obj_tcp_sim.coordinates import wire_orientation
class Tests(unittest.TestCase):
 def test_original_conversion_and_body_axes(self):
  q=wire_orientation((0.,0.,0.,1.))
  p,r=converted_pose((10.,-20.,3.),q)
  self.assertEqual(p,(10.,20.,3.))
  self.assertAlmostEqual(abs(r[3]),1.)
 def test_anchor_and_world_rotation_do_not_rotate_body_velocity(self):
  a=Alignment((10.,20.,3.),(0.,0.,0.,1.),(1.,2.,4.),(0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)))
  p,q=a.apply((11.,20.,3.),(0.,0.,0.,1.))
  self.assertAlmostEqual(p[0],1.);self.assertAlmostEqual(p[1],3.);self.assertAlmostEqual(p[2],4.)
  rates=PoseRates();rates.update((10.,20.,3.),(0.,0.,0.,1.),1.)
  v,w=rates.update((10.02,20.,3.),(0.,0.,0.,1.),1.1)
  self.assertAlmostEqual(v[0],.2);self.assertAlmostEqual(v[1],0.)
 def test_gap_is_unknown_not_stopped(self):
  rates=PoseRates();self.assertIsNone(rates.update((0,0,0),(0,0,0,1),1.))
  self.assertIsNone(rates.update((0,0,0),(0,0,0,1),2.))
 def test_invalid_quaternion_rejected(self):
  with self.assertRaises(ValueError):converted_pose((0,0,0),(0,0,0,0))
if __name__=='__main__':unittest.main()
