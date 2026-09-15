"""Valid bounded records and actual current owned-process resource observations."""
import json
import os
import pytest
from lunar_drl_exploration.metrics import MetricsWriter, ResourceMonitor, ResourceLimitError


def test_metrics_keep_complete_recent_json_records_with_fixed_bound(tmp_path):
    path=tmp_path/'metrics.jsonl'; writer=MetricsWriter(path,max_bytes=256)
    for step in range(100): writer.append({'step':step,'loss':.25})
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]['step']==99 and len(records)>1 and path.stat().st_size<=256
    assert all(b['step']==a['step']+1 for a,b in zip(records,records[1:]))
    before=path.read_bytes()
    with pytest.raises(ValueError): writer.append({'loss':float('nan')})
    with pytest.raises(ValueError): writer.append({'message':'x'*1000})
    assert path.read_bytes()==before
    with path.open('ab') as stream: stream.write(b'{"broken":')
    writer=MetricsWriter(path,max_bytes=256)
    writer.append({'step':100})
    assert all(isinstance(json.loads(line),dict) for line in path.read_text().splitlines())
    assert json.loads(path.read_text().splitlines()[-1])['step']==100


def test_proc_accounting_counts_owned_pss_once_ignores_peak_rss_and_reports_observed(tmp_path):
    proc=tmp_path/'proc'; proc.mkdir()
    (proc/'meminfo').write_text('MemTotal: 10000 kB\nMemAvailable: 3000 kB\n')
    for pid,pss in ((11,100),(12,200)):
        (proc/str(pid)).mkdir()
        (proc/str(pid)/'smaps_rollup').write_text(f'Rss: 9000 kB\nPss: {pss} kB\n')
    monitor=ResourceMonitor([11,12,11],proc_root=proc,system_reserve_bytes=2048*1024)
    snapshot=monitor.observe(tmp_path)
    assert snapshot['owned_pss_bytes']==300*1024 and snapshot['system_available_bytes']==3000*1024
    monitor.check(snapshot)
    low=dict(snapshot,system_available_bytes=1000)
    with pytest.raises(ResourceLimitError,match='system reserve') as error: monitor.check(low)
    assert error.value.observed==1000 and error.value.limit==2048*1024
    monitor=ResourceMonitor([11,12],proc_root=proc,owned_pss_limit_bytes=250*1024,system_reserve_bytes=0)
    with pytest.raises(ResourceLimitError,match='owned PSS'): monitor.check(snapshot)
    # Actual Linux observation is fresh and includes only explicitly owned PIDs.
    real=ResourceMonitor([os.getpid()],system_reserve_bytes=0).observe(tmp_path)
    assert real['owned_pss_bytes']>0 and real['system_available_bytes']>0 and real['disk_free_bytes']>0


def test_metrics_share_total_artifact_budget_and_valid_output_survives_rejection(tmp_path):
    path=tmp_path/'metrics.jsonl'
    writer=MetricsWriter(path,max_bytes=256,total_max_bytes=100)
    writer.append({'step':0})
    (tmp_path/'resume.pt').write_bytes(b'x'*85)
    before=path.read_bytes()
    with pytest.raises(ResourceLimitError,match='artifact'): writer.append({'step':1})
    assert path.read_bytes()==before
