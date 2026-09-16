#!/usr/bin/env python3
"""Export TCP runtime sources and preprocessed terrain, using the Orin 1.0 layout."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'tools/tcp_deployment'


def export(destination, terrain, allow_dirty=False):
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).decode()
    if status and not allow_dirty:
        raise SystemExit('Source is dirty; commit reviewed changes or use --allow-dirty for validation only')
    archive = destination.with_name(destination.name+'.tar.gz')
    if archive.exists() or archive.with_name(archive.name+'.sha256').exists():
        raise SystemExit(f'Refusing to replace existing archive/checksum: {archive}')
    required = ('metadata.json', 'elevation.npy', 'sample_xy.npy', 'triangles.bin', 'index_ids.npy', 'index_offsets.npy')
    for name in required:
        if not (terrain / name).is_file():
            raise SystemExit(f'Missing terrain input: {terrain / name}')
    metadata = json.loads((terrain / 'metadata.json').read_text())
    destination.mkdir(parents=True, exist_ok=False)
    def copy(relative):
        source = TEMPLATES / 'templates' / relative
        if not source.is_file():
            source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in (TEMPLATES / 'runtime-files.txt').read_text().splitlines():
        copy(relative)
    # Keep controller compatibility messages without exporting the old planner.
    for package in ('lunar_planning_msgs', 'lunar_pure_wheeled_controller', 'lunar_obj_tcp_sim'):
        for source in sorted((ROOT / 'ros2_ws/src' / package).rglob('*')):
            if not source.is_file() or any(p in ('test', 'tests', '__pycache__', '.pytest_cache') or p.endswith('.egg-info') for p in source.parts):
                continue
            if source.suffix in ('.pyc', '.pyo'):
                continue
            copy(str(source.relative_to(ROOT)))
    for source in sorted((ROOT / 'scripts/simulation').glob('*.sh')):
        copy(str(source.relative_to(ROOT)))
    for name in ('OBJ_TCP数字仿真使用说明.md', 'OBJ_TCP数字仿真验证记录.md', 'TCP仿真部署说明.md'):
        copy('docs/' + name)
    shutil.copy2(ROOT / 'docs/TCP仿真部署说明.md', destination / 'README.md')
    for source in (TEMPLATES / 'scripts').glob('*.sh'):
        target = destination / 'scripts' / source.name
        shutil.copy2(source, target)
        target.chmod(0o755)
    (destination / 'maps/terrain').mkdir(parents=True)
    for name in required:
        shutil.copy2(terrain / name, destination / 'maps/terrain' / name)
    source_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    provenance = {'version': 'P4', 'source_commit': source_hash,
                  'source_dirty': bool(status), 'reference_deployment_commit': (TEMPLATES/'reference.txt').read_text().strip(),
                  'map_source_name': terrain.name, 'map_alignment': metadata.get('alignment_status', 'NOT_EXTERNALLY_ALIGNED'),
                  'runtime_packages': sorted(p.name for p in (destination/'ros2_ws/src').iterdir())}
    (destination/'ORIGIN.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    (destination/'VERSION.txt').write_text('P4\n')
    checksums = []
    for p in sorted(destination.rglob('*')):
        if p.is_file():
            checksums.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(destination)))
    (destination/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
    archive = destination.with_name(destination.name+'.tar.gz')
    if archive.exists():
        raise SystemExit(f'Refusing to replace {archive}')
    with tarfile.open(archive, 'w:gz') as output:
        output.add(destination, arcname=destination.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'directory':str(destination), 'archive':str(archive), 'files':len(checksums)+1, 'bytes':archive.stat().st_size, 'sha256':digest}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args()
    export(args.output.resolve(), args.map.resolve(), args.allow_dirty)
