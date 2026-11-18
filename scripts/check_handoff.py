"""用宿主 Python 标准库检查源码、模型、冻结地图和初始化版本，不安装依赖。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    official=json.loads((ROOT/'assets/OFFICIAL_MODELS.json').read_text())
    actual=subprocess.check_output(['git','-C',str(ROOT/'upstream/REASAN'),'rev-parse','HEAD'],text=True).strip()
    if actual!=official['commit']:raise RuntimeError('官方源码提交与固定版本不同')
    checks={}
    for path,record in official['files'].items():checks['upstream/REASAN/'+path]=record['sha256']
    for path,record in json.loads((ROOT/'assets/a2/SOURCE.json').read_text())['files'].items():checks[path]=record['sha256']
    for path,h in json.loads((ROOT/'assets/a2/USD_SOURCE_v1.json').read_text())['files'].items():checks['assets/a2/usd_v1/'+path]=h
    for path,h in json.loads((ROOT/'assets/isaac45/SOURCE.json').read_text())['files_sha256'].items():checks['assets/isaac45/'+path]=h
    checks.update(json.loads((ROOT/'assets/a2/spawn_geometry_v1/SOURCE.json').read_text())['files_sha256'])
    terrain=json.loads((ROOT/'assets/evaluation/public_static_v1/manifest.json').read_text())
    checks['assets/evaluation/public_static_v1/terrain.npz']=terrain['terrain_sha256']
    for path,expected in checks.items():
        if digest(ROOT/path)!=expected:raise RuntimeError('文件与来源不符: '+path)
    result={'passed':True,'checked_files':len(checks),'upstream_commit':actual,
        'scope':'source, pretrained models, robot assets and frozen evaluation protocol; no simulation or training',
        'a2_result_interpretation':'zero-shot transfer of frozen Go2 navigation and safety policies to A2 MoE'}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
