import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
from lunar_drl_exploration.processes import OwnedProcesses, ros_arguments


def test_owned_failure_retains_bounded_error_and_reaps():
    owner=OwnedProcesses()
    p=owner.spawn([sys.executable,'-c',"import sys;sys.stderr.write('x'*20000+'EXACT_REASON');sys.exit(7)"])
    p.wait(timeout=5)
    with pytest.raises(RuntimeError, match='EXACT_REASON'): owner.check()
    assert len(owner.error_tail()) <= 8192
    owner.close()
    assert not owner.pids


def test_forced_worker_death_kills_only_owned_child(tmp_path):
    code='''import sys,time
from lunar_drl_exploration.processes import OwnedProcesses
p=OwnedProcesses(); c=p.spawn([sys.executable,'-c','import time;time.sleep(60)'])
print(c.pid,flush=True)
time.sleep(60)
'''
    env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH',''))
    worker=subprocess.Popen([sys.executable,'-c',code], stdout=subprocess.PIPE,text=True,env=env)
    child=int(worker.stdout.readline())
    worker.kill(); worker.wait(timeout=5)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        path=Path(f'/proc/{child}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z': break
        time.sleep(.02)
    else: pytest.fail('owned child survived forced worker death')


def test_ros_topics_cannot_escape_training_namespace():
    with pytest.raises(ValueError): ros_arguments('/lunar_training/env_0', {'command_topic':'/Car/T5/Car_Cmd_Vel'})


def test_close_reaps_owned_live_process_and_preserves_unowned():
    outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    owner=OwnedProcesses()
    child=owner.spawn([sys.executable,'-c','import time;time.sleep(60)'])
    try:
        owner.close()
        assert child.poll() is not None and not owner.pids
        assert outsider.poll() is None
    finally:
        owner.close();outsider.terminate();outsider.wait(timeout=5)
