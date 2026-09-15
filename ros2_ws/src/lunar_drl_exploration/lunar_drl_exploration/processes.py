"""Exact owned children, Linux parent-death binding and bounded in-memory errors."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Run prctl in a fresh interpreter, not preexec_fn in a multithreaded ROS worker.
# Parent-race check closes the death-before-prctl window. exec preserves PDEATHSIG.
_LAUNCH = '''import ctypes,os,signal,sys
parent=int(sys.argv[1])
if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0): raise OSError('prctl PDEATHSIG')
if os.getppid()!=parent: os._exit(125)
os.execvpe(sys.argv[2],sys.argv[2:],os.environ)
'''


def ros_arguments(namespace, parameters):
    if not namespace.startswith('/lunar_training/') or '/..' in namespace:
        raise ValueError('simulation namespace required')
    result=['--ros-args','-r',f'__ns:={namespace}','--log-level','error',
            '--disable-rosout-logs','--disable-external-lib-logs']
    for key,value in parameters.items():
        if ('topic' in key or key in ('action_name','navigation_action','policy_map_service')) and not str(value).startswith(namespace+'/'):
            raise ValueError(f'{key} must be inside the simulation namespace')
        result.extend(['-p',f'{key}:={str(value).lower() if isinstance(value,bool) else value}'])
    return result


class OwnedProcesses:
    def __init__(self):
        self.children=[]

    @property
    def pids(self): return tuple(p.pid for p,_ in self.children)

    def spawn(self, command, env=None):
        p=subprocess.Popen([sys.executable,'-c',_LAUNCH,str(os.getpid()),*map(str,command)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
            start_new_session=True,env=env)
        os.set_blocking(p.stderr.fileno(),False)
        self.children.append((p,bytearray()))
        return p

    def start(self, package, executable, namespace, parameters, env=None):
        from ament_index_python.packages import get_package_prefix
        path=Path(get_package_prefix(package))/'lib'/package/executable
        if not path.is_file(): raise RuntimeError(f'ROS executable not built: {path}')
        return self.spawn([str(path),*ros_arguments(namespace,parameters)],env=env).pid

    def _drain(self):
        for p,tail in self.children:
            while True:
                chunk=p.stderr.read(8192)
                if not chunk: break
                tail.extend(chunk)
                del tail[:-4096]

    def error_tail(self):
        self._drain()
        return '\n'.join(t.decode(errors='replace') for _,t in self.children)[-8192:]

    def check(self):
        self._drain()
        for p,tail in self.children:
            if p.poll() is not None:
                raise RuntimeError(f'owned ROS process {p.pid} exited ({p.returncode}): '+tail.decode(errors='replace'))

    def close(self):
        for p,_ in reversed(self.children):
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        deadline=time.monotonic()+3
        for p,_ in reversed(self.children):
            try: p.wait(timeout=max(.01,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
            p.stderr.close()
        self.children.clear()
