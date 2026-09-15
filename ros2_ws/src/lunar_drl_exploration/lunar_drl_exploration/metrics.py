"""Bounded JSONL and current Linux PSS/system-reserve observations, without psutil."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


class ResourceLimitError(RuntimeError):
    def __init__(self, reason, observed, limit):
        self.reason, self.observed, self.limit = reason, observed, limit
        super().__init__(f'{reason}: observed={observed}, limit={limit}')


def artifact_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob('*') if item.is_file())


class MetricsWriter:
    """One bounded file, whole recent records; incomplete final writes are repaired."""
    def __init__(self, path, max_bytes=8*1024**2, *, total_max_bytes=20*1024**3):
        self.path=Path(path); self.max_bytes=int(max_bytes); self.total_max_bytes=total_max_bytes
        if self.max_bytes < 3: raise ValueError('metrics budget too small')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():
            # Read at most the cap; drop a leading fragment and malformed tail.
            with self.path.open('rb') as stream:
                size=stream.seek(0,2); offset=max(0,size-self.max_bytes); stream.seek(offset)
                data=stream.read(self.max_bytes)
            if offset: data=data.partition(b'\n')[2]
            lines=[]
            for line in data.splitlines(keepends=True):
                try:
                    if not line.endswith(b'\n') or not isinstance(json.loads(line),dict): break
                except (ValueError,UnicodeError): break
                lines.append(line)
            with self.path.open('wb') as stream: stream.write(b''.join(lines))

    def append(self, record):
        if not isinstance(record,dict): raise ValueError('metric must be an object')
        line=(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')
        if len(line)>self.max_bytes: raise ValueError('metric record exceeds byte budget')
        size=self.path.stat().st_size if self.path.exists() else 0
        if size+len(line)>self.max_bytes:
            # In-place bounded compaction: no unaccounted second metrics artifact.
            data=self.path.read_bytes()
            start=max(0,len(data)+len(line)-self.max_bytes)
            if start: start=data.index(b'\n',start-1)+1
            output=data[start:]+line
            self._check_total(len(output)-size)
            with self.path.open('wb') as stream: stream.write(output)
        else:
            self._check_total(len(line))
            with self.path.open('ab') as stream: stream.write(line)

    def _check_total(self, increase):
        observed=artifact_bytes(self.path.parent)+increase
        if observed>self.total_max_bytes:
            raise ResourceLimitError('artifact metrics bytes',observed,self.total_max_bytes)


class ResourceMonitor:
    def __init__(self, owned_pids, *, system_reserve_bytes=2*1024**3,
                 owned_pss_limit_bytes=None, proc_root='/proc'):
        self.owned_pids=tuple(sorted(set(int(pid) for pid in owned_pids)))
        if not self.owned_pids: raise ValueError('explicit owned process IDs required')
        self.system_reserve_bytes=system_reserve_bytes
        self.owned_pss_limit_bytes=owned_pss_limit_bytes
        self.proc_root=Path(proc_root)

    @staticmethod
    def _kb_fields(path):
        result={}
        for line in path.read_text().splitlines():
            if ':' in line:
                key,value=line.split(':',1)
                parts=value.split()
                if parts and parts[0].isdigit(): result[key]=int(parts[0])*1024
        return result

    def observe(self, output_dir):
        pss={}
        for pid in self.owned_pids:
            # Missing/unreadable live ownership data is explicit, never silently 0.
            pss[str(pid)]=self._kb_fields(self.proc_root/str(pid)/'smaps_rollup')['Pss']
        memory=self._kb_fields(self.proc_root/'meminfo')
        return dict(monotonic_s=time.monotonic(),owned_pss_by_pid=pss,
            owned_pss_bytes=sum(pss.values()),system_available_bytes=memory['MemAvailable'],
            system_total_bytes=memory['MemTotal'],disk_free_bytes=shutil.disk_usage(output_dir).free,
            output_bytes=artifact_bytes(output_dir))

    def check(self, snapshot):
        if snapshot['system_available_bytes'] < self.system_reserve_bytes:
            raise ResourceLimitError('system reserve',snapshot['system_available_bytes'],self.system_reserve_bytes)
        if self.owned_pss_limit_bytes is not None and snapshot['owned_pss_bytes']>self.owned_pss_limit_bytes:
            raise ResourceLimitError('owned PSS',snapshot['owned_pss_bytes'],self.owned_pss_limit_bytes)


def live_hardware(output_dir):
    """Record available hardware, not an admission gate; no Torch import in workers."""
    memory=ResourceMonitor._kb_fields(Path('/proc/meminfo'))
    result=dict(cpu_count=os.cpu_count(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        memory_total_bytes=memory['MemTotal'],memory_available_bytes=memory['MemAvailable'],
        disk_free_bytes=shutil.disk_usage(output_dir).free,gpu=[])
    try:
        probe=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        for line in probe.stdout.splitlines():
            name,total,free=line.rsplit(',',2)
            result['gpu'].append(dict(name=name.strip(),total_bytes=int(total)*1024**2,free_bytes=int(free)*1024**2))
    except (OSError,subprocess.SubprocessError,ValueError) as error:
        result['gpu_observation_error']=str(error)
    return result
