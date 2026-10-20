"""Sample the admitted physical host with its own wall clock.

[Role] Feed actual npu-smi samples into the existing npu-top SQLite/window APIs.
[Invariants] No local/remote clock assumption; missing throttle registers stay null.
[Timeouts & Locks] 5s cadence; inner 5+5+4s, outer 20s; independent SQLite file.
[Side-Effect Policy] Read-only host observation and task-local evidence writes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mindie_coordinator.host_queue import HostQueue
from mindie_coordinator.host.mindie_npu_coordination import parse_npu_smi_info as parse_occupancy
from mindie_coordinator.npu_top_bridge import observed_process_interference
from npu_top import dcmi_telemetry
from npu_top.db import Database
from npu_top.npu_smi import parse_npu_common, parse_npu_smi_info
from remote_dev.core.endpoint import resolve_endpoint
from remote_dev.core.ssh_transport import run_remote_python


SAMPLE = r'''
request=json.load(sys.stdin)
wall_before_ns=time.time_ns()
perf_ns=time.perf_counter_ns()
mono_ns=time.monotonic_ns()
wall_after_ns=time.time_ns()
clock_anchor={'wall_before_ns':wall_before_ns,'perf_counter_ns':perf_ns,
    'monotonic_ns':mono_ns,'wall_after_ns':wall_after_ns}
offsets=Path('/proc/self/timens_offsets')
clock_anchor['timens_offsets']=offsets.read_text() if offsets.exists() else None
started=time.time()
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=5)
if info.returncode: raise RuntimeError(info.stderr)
common=subprocess.run(['npu-smi','info','-t','common','-i',str(request['card'])],capture_output=True,text=True,timeout=5)
try:
    probe=subprocess.run([sys.executable,'-','--cards',str(request['card'])],input=request['dcmi_source'],capture_output=True,text=True,timeout=4)
    if probe.returncode: raise RuntimeError(probe.stderr)
    dcmi=json.loads(probe.stdout)
except (subprocess.TimeoutExpired,RuntimeError,ValueError) as exc:
    dcmi={'state':'unavailable','error':str(exc),'error_type':type(exc).__name__}
parsed=parse_npu_smi_info(info.stdout)
marker=request.get('marker')
identities={}
for reason in (parsed.get('busy') or {}).get(str(request['card']),[]):
    if reason.get('kind')!='process': continue
    pid=reason['pid']; root=Path('/proc')/str(pid)
    try:
        before=(root/'stat').read_text().rsplit(') ',1)[1].split()[19]
        tokens=(root/'environ').read_bytes().split(b'\0')
        after=(root/'stat').read_text().rsplit(') ',1)[1].split()[19]
        owned=(before==after and marker is not None and ('REMOTE_DEV_JOB_TOKEN='+marker).encode() in tokens)
        identities[str(pid)]={'start_ticks':after,'owned':owned,'identity_stable':before==after}
    except (FileNotFoundError,ProcessLookupError):
        identities[str(pid)]={'owned':None,'identity_stable':False}
print(json.dumps({'started_at':started,'finished_at':time.time(),'info':info.stdout,
    'common':common.stdout,'common_exit_code':common.returncode,'common_stderr':common.stderr,
    'clock_anchor':clock_anchor,'dcmi':dcmi,'process_ownership':identities,
    'marker_known':marker is not None,'occupancy_status':parsed.get('status')}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--interval', type=int, default=5)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    database=Database(args.output/'telemetry.sqlite3')
    database.initialize()
    record=json.loads(args.record.read_text())
    host=record['host_endpoint']
    server=database.upsert_server({'id':'stage1-host','name':'stage1-host','host':host['host'],
        'port':host['port'],'username':host['user'],'enabled':True,'tags':['live-stage1']})
    source=HostQueue().source()+'\nimport sys\n'+SAMPLE
    dcmi_source=Path(dcmi_telemetry.__file__).read_text()
    deadline=time.monotonic()+args.seconds
    next_sample=time.monotonic()
    number=0
    while time.monotonic()<deadline:
        record=json.loads(args.record.read_text())
        marker=(record.get('remote',{}).get('receipt',{}).get('process_guard') or {}).get('marker')
        sample=run_remote_python(resolve_endpoint({**host,'root':'/','cwd':'/','runtime_env':False}),
            source,{'card':record['devices'][0],'marker':marker,'dcmi_source':dcmi_source},timeout_ms=20000)
        if 'info' not in sample:
            raise RuntimeError('host sampling failed: '+str(sample))
        number+=1
        sample['run_id']=record['run_id']
        sample['planned_interval_seconds']=args.interval
        with (args.output/'raw-samples.jsonl').open('a') as stream:
            stream.write(json.dumps(sample)+'\n')
        parsed=parse_npu_smi_info(sample['info'])
        occupancy=parse_occupancy(sample['info'])
        frequencies=parse_npu_common(sample['common']) if sample['common_exit_code']==0 else {}
        for device in parsed['devices']:
            freq=frequencies.get(device['npu_id'])
            if freq:
                device['freq_mhz']=freq['freq_mhz']
            dcmi=(sample.get('dcmi',{}).get('cards') or {}).get(str(device['npu_id']))
            if dcmi:
                device['dcmi_freq_reduc']=dcmi
            if device['npu_id'] in record['devices']:
                device['resource_interference_detected']=observed_process_interference(
                    occupancy, device['npu_id'], sample['process_ownership'],
                    marker_known=sample['marker_known'])
        snapshot={**parsed,'collected_at':int(sample['finished_at'])}
        database.record_success(server['id'],snapshot,persist_sample=True)
        if (args.output/'stop').exists():
            break
        next_sample+=args.interval
        time.sleep(max(0,next_sample-time.monotonic()))
    (args.output/'collector.json').write_text(json.dumps({'samples':number,
        'host':host['host'],'devices':record['devices'],'planned_interval_seconds':args.interval,
        'clock_basis':'remote physical host time.time; same kernel wall clock as container process',
        'database':str(args.output/'telemetry.sqlite3')},indent=2))
    print(json.dumps({'samples':number,'output':str(args.output)}))


if __name__=='__main__':
    main()
