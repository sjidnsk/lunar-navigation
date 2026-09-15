"""Opt-in actual isolated Jazzy action/controller/plant integration (no mocks)."""
import os
import math
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('LUNAR_DRL_RUN_ROS')!='1',reason='actual isolated ROS opt-in')


@pytest.mark.parametrize('start_xy',[.1,.199])
def test_native_cold_start_spin_adjacent_arrival_and_frozen_transition(start_xy):
    from lunar_drl_exploration.config import TrainingConfig
    from lunar_drl_exploration.contracts import Pose,TaskSpec
    from lunar_drl_exploration.scene import TerrainGrid
    from lunar_drl_exploration.ros_env import RosExplorationEnv
    config=TrainingConfig(target_rtf=10.)
    heights=np.zeros((60,60),np.float32);heights[:,26:]=2.
    terrain=TerrainGrid.from_heights(heights,.2,(-4.,-4.),config.platform)
    task=TaskSpec('native_integration','map',np.array([[-5,-5],[8,-5],[8,8],[-5,8]]))
    env=RosExplorationEnv(config,0,domain_base=200,wall_timeout_s=30.)
    try:
        observation,privileged=env.reset_scene(terrain,Pose(start_xy,start_xy,0.),task)
        assert env.node.count_publishers(env.namespace+'/cmd_vel')==1
        assert env.adapter.snapshot.goal_position_tolerance_m==.05
        assert env.episode_metadata['initialization_turn_rad']>0
        assert env.plant.distance_m==0
        assert env.plant.pose.x==start_xy and env.plant.pose.y==start_xy
        for goal in [(-.1,.1,0.),(-.3,.1,0.),(-.3,.1,math.pi/2)]:
            result=env._execute_goal(goal)
            assert result.outcome==0 and result.reason_code=='GOAL_REACHED'
            env._snapshot()
            assert terrain.world_to_cell(*goal[:2])==terrain.world_to_cell(env.plant.pose.x,env.plant.pose.y)
            assert math.hypot(env.plant.pose.x-goal[0],env.plant.pose.y-goal[1])<=.05
        env.observation,env.report=env.core.observe(env._snapshot(),env.adapter.velocity)
        env.privileged=env._privileged();env.actor_version=19;env.episode_budget=1
        indices=np.flatnonzero(np.linalg.norm(env.observation.goals[:,:2]-[env.plant.pose.x,env.plant.pose.y],axis=1)>.1)
        assert len(indices)>0
        index=int(indices[np.argmin(np.linalg.norm(env.observation.goals[indices,:2]-[env.plant.pose.x,env.plant.pose.y],axis=1))])
        transition=env.step(index)
        assert tuple(transition.observation.goals[index])==env.last_execution.goal
        assert transition.actor_version==19 and transition.truncated and not transition.terminated
        assert env.last_execution.goal_cell==env.last_execution.actual_cell
        assert env.last_execution.reason_code=='GOAL_REACHED'
        assert transition.parts.distance_m>0 and transition.parts.turn_rad>=0
        assert env.progress()['max_command_speed_mps']<=.2+1e-9
        evidence=list(env.adapter.planning_evidence)
        plans=[d for d in evidence if d.get('cycle_result')=='PLAN_FOUND']
        assert plans and any(d.get('cycle_result')=='GOAL_REACHED' for d in evidence)
        for plan in plans:
            assert any(ref['session_id']==plan['session_id'] and ref['active'] and ref['segment_revision']==int(plan['segment_revision']) and
                ref['fine_revision']==int(plan['fine_traversability_revision']) for ref in env.adapter.references)
        assert env.progress()['sensor_frames']>20
        assert env.reference.covered_area(transition.next_privileged.observed)<=env.reference.area_m2+1e-8
    finally:env.close()
    assert not env.owned_pids
