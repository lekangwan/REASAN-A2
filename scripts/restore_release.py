"""从已校验的附件恢复运行资产及可选结果；已有的不同内容不会被覆盖。"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,default=ROOT/'release-downloads')
    parser.add_argument('--with-results',action='store_true')
    parser.add_argument('--with-videos',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'docs/RELEASE_ARTIFACTS.json').read_text())
    for artifact in manifest['artifacts']:
        if artifact['kind']=='results' and not args.with_results:continue
        if artifact['kind']=='videos' and not args.with_videos:continue
        path=args.bundle_dir/artifact['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:raise RuntimeError('附件校验失败: '+path.name)
        with tarfile.open(path,'r:gz') as archive:
            members=archive.getmembers()
            # 在写文件前检查全部路径、文件类型和已存在的内容。
            for member in members:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or name.parts[0] not in ('assets','results') or not member.isfile():
                    raise RuntimeError('附件包含不允许的路径或文件类型: '+member.name)
                target=ROOT/member.name
                try:target.resolve().relative_to(ROOT.resolve())
                except ValueError:raise RuntimeError('附件路径越界')
                if target.exists():
                    with archive.extractfile(member) as f:expected=hashlib.sha256(f.read()).hexdigest()
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise RuntimeError('现有文件内容不同，停止覆盖: '+str(target))
            for member in members:
                target=ROOT/member.name
                if target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
                target.chmod(member.mode&0o777)
        print('已恢复 '+path.name,flush=True)


if __name__=='__main__':main()
