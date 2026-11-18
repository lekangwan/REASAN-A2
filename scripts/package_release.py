"""为固定版本生成可校验的资产、结果和视频附件；不包含缓存或 NVIDIA 资产。"""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
TAG='v0.1.0'


def main():
    out=ROOT/'release/artifacts';out.mkdir(parents=True,exist_ok=True)
    assets=[p for prefix in ('assets/a2','assets/evaluation') for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    results=[p for prefix in ('results/evaluation','results/debug') for p in (ROOT/prefix).rglob('*') if p.is_file()]
    groups={'assets':assets,'results':[p for p in results if p.suffix!='.mp4'],
            'videos':[p for p in results if p.suffix=='.mp4']}
    artifacts=[]
    for kind,files in groups.items():
        name='reasan-a2-'+kind+'-'+TAG+'.tar.gz';path=out/name
        with path.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1 if kind=='videos' else 6) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
            for file in sorted(files):
                info=archive.gettarinfo(str(file),arcname=str(file.relative_to(ROOT)))
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                with file.open('rb') as data:archive.addfile(info,data)
        record={'kind':kind,'name':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'files':len(files)}
        artifacts.append(record);print(json.dumps(record),flush=True)
    manifest={'repository':'lekangwan/REASAN-A2','tag':TAG,'artifacts':artifacts,
        'nvidia_assets':'downloaded separately from official NVIDIA source with assets/isaac45/SOURCE.json',
        'source_commit':'655bf4f1cc23e835d87bf9a887972af6760af162'}
    (ROOT/'docs/RELEASE_ARTIFACTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in artifacts))


if __name__=='__main__':main()
