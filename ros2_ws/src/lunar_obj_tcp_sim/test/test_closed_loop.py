"""Opt-in ROS launch + real TCP loopback E2E, explicitly not a UE physics test."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

pytestmark=pytest.mark.skipif(os.environ.get('LUNAR_RUN_TCP_E2E')!='1', reason='set LUNAR_RUN_TCP_E2E=1 after building the ROS overlay')


def spin_until(executor,predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        executor.spin_once(timeout_sec=.02)
        if predicate(): return
    raise AssertionError('Timed out waiting for '+getattr(predicate,'__name__','condition'))


@pytest.mark.parametrize('mode',['nav','explore','real','obstacle','blind','slip'])
def test_formal_closed_loop_through_tcp(tmp_path,mode):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from diagnostic_msgs.msg import DiagnosticArray
    from std_msgs.msg import String
    from lunar_planning_msgs.action import NavigateToPose
    from lunar_planning_msgs.msg import PathReference,TrackingStatus
    from lunar_pure_exploration_msgs.msg import PureExplorationStatus
    from lunar_obj_tcp_sim.prepare import prepare_map
    from loopback_vehicle import LoopbackVehicle

    if mode == 'real':
        if not os.environ.get('LUNAR_REAL_OBJ_MAP'):
            pytest.skip('set LUNAR_REAL_OBJ_MAP to test a prepared source map')
        from lunar_obj_tcp_sim.geometry import TerrainMap
        from ament_index_python.packages import get_package_share_directory
        import yaml
        map_dir=Path(os.environ['LUNAR_REAL_OBJ_MAP'])
        terrain=TerrainMap(map_dir)
        xmin,ymin,xmax,ymax=terrain.metadata['task_bounds_xy']
        start_x,start_y=(xmin+xmax)/2,(ymin+ymax)/2
        position=(start_x,start_y,terrain.height_at(start_x,start_y))
        config=yaml.safe_load((Path(get_package_share_directory('lunar_obj_tcp_sim'))/'config/simulation.yaml').read_text())
        config['simulation'].update(auto_start=False,initial_scan=False)
        config_file=tmp_path/'real-config.yaml'; config_file.write_text(yaml.safe_dump(config))
    else:
        obj=tmp_path/'ground.obj'
        obj.write_text('g ground\nv -22 -22 0\nv 22 -22 0\nv 22 22 0\nv -22 22 0\nf 1 2 3\nf 1 3 4\n')
        if mode=='obstacle':
            with obj.open('a') as stream:
                stream.write('g rock\nv -.3 -.8 1.5\nv .3 -.8 1.5\nv .3 .8 1.5\nv -.3 .8 1.5\nf 5 6 7\nf 5 7 8\n')
        map_dir=tmp_path/'map'
        prepare_map(obj,map_dir,center=(0,0),size=8,resolution=.2,halo=12,scale=1,axes='x,y,z')
        position=(-2.,0.,0.) if mode=='obstacle' else (0.,0.,0.)
    if mode == 'blind':
        import yaml
        config=yaml.safe_load((Path(__file__).parents[1]/'config/simulation.yaml').read_text())
        config['simulation']['near_field_radius_m']=1.4
        config_file=tmp_path/'blind-config.yaml';config_file.write_text(yaml.safe_dump(config))
    vehicle=LoopbackVehicle(position=position,slip_mps=.04 if mode=='slip' else 0.)
    output=tmp_path/'launch.log'; log=output.open('w')
    launch_mode='explore' if mode in ('real','blind','slip') else ('nav' if mode=='obstacle' else mode)
    command=['python3','-m','lunar_obj_tcp_sim.operator','--prefix','/lunar_sim','--mode',launch_mode,'--',
        'ros2','launch','lunar_obj_tcp_sim','obj_tcp.launch.py',
        'host:=127.0.0.1',f'port:={vehicle.port}',f'map_directory:={map_dir}',
        f'mode:={launch_mode}','start_rviz:=false','start_local_rviz:=false']
    if mode in ('real','blind'):command.append(f'config:={config_file}')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init(); node=Node('tcp_closed_loop_probe'); executor=SingleThreadedExecutor();executor.add_node(node)
    status={}; diagnostics=[]; references=[]; tracking=[]; exploration=[]
    node.create_subscription(String,'/lunar_sim/observation_status',lambda m:status.update(json.loads(m.data)),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(DiagnosticArray,'/lunar_sim/planning/diagnostics',diagnostics.append,10)
    node.create_subscription(PathReference,'/lunar_sim/path_reference',references.append,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node.create_subscription(TrackingStatus,'/lunar_sim/tracking_status',tracking.append,10)
    node.create_subscription(PureExplorationStatus,'/lunar_sim/exploration/status',exploration.append,10)
    from visualization_msgs.msg import MarkerArray
    markers=[]
    for topic in ('observed_cells','display','hud_local','terrain_preview'):
        node.create_subscription(MarkerArray,'/lunar_sim/'+topic,markers.append,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    client=ActionClient(node,NavigateToPose,'/lunar_sim/navigate_to_pose')
    results=[]
    try:
        if mode == 'blind':
            spin_until(executor,lambda:status.get('phase')=='BOOTSTRAP_FAILED',15)
            initial_samples=status['samples']
            spin_until(executor,lambda:status['samples']>=initial_samples+5,10)
            assert status['phase']=='BOOTSTRAP_FAILED'
            assert not any(r.state==r.ACTIVE for r in references)
            assert all(v==0 and w==0 for v,w in vehicle.commands)
            assert not vehicle.errors
            (tmp_path/'receipt.json').write_text(json.dumps(dict(
                mode=mode, phase=status['phase'], samples_before=initial_samples,
                samples_after=status['samples'], continued_observation=True, no_motion=True)))
            return
        if mode in ('nav','real','obstacle'):
            if mode in ('nav','obstacle'):
                spin_until(executor,lambda:status.get('phase')=='KNOWN_MAP_READY',45)
                goals=[(2.,0.,0.)] if mode=='obstacle' else [(1.,0.,0.),(1.,0.,.8),(.2,0.,0.)]
            else:
                spin_until(executor,lambda:status.get('samples',0)>=5,20)
                goals=[(position[0]+1.,position[1],0.)]
            for x,y,yaw in goals:
                future=client.send_goal_async(NavigateToPose.Goal(target_x_m=x,target_y_m=y,has_target_yaw=True,target_yaw_rad=yaw))
                spin_until(executor,future.done,5); handle=future.result(); assert handle.accepted
                future=handle.get_result_async(); spin_until(executor,future.done,90 if mode=='obstacle' else 45)
                result=future.result().result; results.append(result.reason_code)
                assert result.outcome==0 and result.reason_code=='GOAL_REACHED'
                if mode == 'nav':
                    assert math.hypot(vehicle.position[0]-x, vehicle.position[1]-y) <= .100001
                    assert abs(math.atan2(math.sin(vehicle.yaw-yaw), math.cos(vehicle.yaw-yaw))) <= .050001
            assert abs(vehicle.position[0]-goals[-1][0])<.25
            assert abs(vehicle.v)<.05
        else:
            spin_until(executor,lambda:status.get('phase')=='EXPLORATION_TASK_STARTED' and any(m.state==PureExplorationStatus.COMPLETED for m in exploration),150)
            assert status['phase']=='EXPLORATION_TASK_STARTED'
            if mode=='slip':
                assert status['scan_reanchors']>0, 'injected slip did not trigger cancellation and reanchor'
            assert status['observation_coverage']>.95
            assert any(m.coverage_ratio>=.99 for m in exploration)
            spin_until(executor,lambda:abs(vehicle.v)<.001 and abs(vehicle.w)<.001,10)
        assert status['phase']!='OBSERVATION_ERROR' and markers
        if mode in ('nav','obstacle','explore'):
            assert any(max(map(abs,a))>.1 for a in vehicle.steering_commands), 'no nonzero steering reached TCP receiver'
        if mode=='obstacle':
            assert max(abs(row[1]) for row in vehicle.trajectory)>1.0, 'did not detour around rock'
            assert all(not(-.7<x<.7 and -1.2<y<1.2) for x,y,z,v,w in vehicle.trajectory), 'feedback trajectory intersected obstacle clearance'
        assert max(abs(row[3]) for row in vehicle.trajectory)<=.200001
        assert not vehicle.errors
        assert vehicle.commands and max(abs(v) for v,w in vehicle.commands)<=.200001
        assert any(r.state==r.ACTIVE and r.path.poses for r in references) and tracking
        values=[{kv.key:kv.value for kv in s.values} for msg in diagnostics for s in msg.status]
        assert any(v.get('cycle_result')=='PLAN_FOUND' for v in values)
        receipt={'mode':mode,'results':results,'position':vehicle.position,'max_command_v':max(abs(v) for v,w in vehicle.commands),
                 'min_command_v':min(v for v,w in vehicle.commands),'max_feedback_v':max(abs(row[3]) for row in vehicle.trajectory),'observation':status,
                 'coarse_coverage':max([m.coverage_ratio for m in exploration] or [0]),
                 'exploration_final_reason':exploration[-1].reason_code if exploration else None,
                 'loopback_only':True,'remote_unreal':'NOT_RUN','log':str(output)}
        (tmp_path/'receipt.json').write_text(json.dumps(receipt,indent=2))
    except BaseException:
        log.flush(); print(output.read_text()[-12000:]); print('last observation',status)
        raise
    finally:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
        log.close(); vehicle.close(); executor.shutdown();node.destroy_node();rclpy.shutdown()

    assert 'Traceback' not in output.read_text(), output.read_text()[-5000:]
    assert 'process has died' not in output.read_text(), output.read_text()[-5000:]
    assert not vehicle.errors
    assert vehicle.commands[-1] == (0.,0.), 'last complete TCP control frame was not zero'
    assert 'WARNING: stop was not confirmed' not in output.read_text(), output.read_text()[-3000:]
