"""Replace the known legacy GUI with the P4 copy; never change P3 files."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = '/home/yanfa/P3/roma_t3_algorithm_bundle_20260825/envx_runtime/install/lunar_car_gui/lib/lunar_car_gui/lunar_car_gui'
TARGET = str(ROOT / 'lunar_car_gui_node.py')

def record(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p/'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'identity': fields[19],
                'argv': (p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None

def main():
    with (ROOT/'gui.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = []
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                r = record(p.name)
                if r and len(r['argv']) > 1 and r['argv'][1] in (ORIGINAL, TARGET):
                    found.append(r)
        if len(found) > 1:
            raise RuntimeError('Multiple vehicle GUIs; refusing ambiguous replacement')
        if found and found[0]['argv'][1] == TARGET:
            print('P4 vehicle GUI already running:', found[0]['pid'])
            return
        if found:
            old = found[0]
            proc = Path('/proc')/str(old['pid'])
            env = dict(v.split('=', 1) for v in (proc/'environ').read_bytes().decode().split('\0') if '=' in v)
            args = old['argv'][2:]
            cwd = os.readlink(proc/'cwd')
            archive = ROOT/'backups'/str(time.time_ns())
            archive.mkdir(parents=True, mode=0o700)
            saved = archive/'original-process.json'
            saved.write_text(json.dumps({**old, 'env': env, 'cwd': cwd}))
            saved.chmod(0o600)
            fd = os.pidfd_open(old['pid'])
            try:
                if record(old['pid']) != old:
                    raise RuntimeError('GUI identity changed; refusing stop')
                signal.pidfd_send_signal(fd, signal.SIGINT)
                deadline = time.monotonic()+15
                while record(old['pid']) == old and time.monotonic() < deadline:
                    time.sleep(.1)
                if record(old['pid']) == old:
                    raise RuntimeError('GUI did not exit; no forced kill')
            finally:
                os.close(fd)
        else:
            env = os.environ.copy()
            env['ROS_DOMAIN_ID'] = env['P4_LEGACY_DOMAIN']
            args = ['--ros-args', '-p', 'map_topic:=/map', '-p', 'path_topic:=/plan']
            cwd = str(ROOT)
        env.pop('PYTHONHOME', None)
        env['PYTHONNOUSERSITE'] = '1'
        with (ROOT/'gui.log').open('a') as log:
            child = subprocess.Popen(['/usr/bin/python3', TARGET, *args], env=env, cwd=cwd,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError('P4 GUI exited; inspect gui.log')
        (ROOT/'runtime.json').write_text(json.dumps({**record(child.pid), 'source_sha256':hashlib.sha256(Path(TARGET).read_bytes()).hexdigest()}, indent=2))
        print('P4 vehicle GUI started:', child.pid)

if __name__ == '__main__':
    main()
