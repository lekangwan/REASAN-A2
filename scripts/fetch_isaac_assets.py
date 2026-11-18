"""从 NVIDIA 官方资源地址下载本项目使用的七个文件，并按原清单校验。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    asset_root=ROOT/'assets/isaac45'
    manifest=json.loads((asset_root/'SOURCE.json').read_text())
    for relative,expected in manifest['files_sha256'].items():
        path=asset_root/relative
        if path.exists():
            if sha256(path)!=expected:raise RuntimeError('现有文件与固定版本不符: '+relative)
        elif args.verify_only:raise FileNotFoundError(path)
        else:
            url=manifest['asset_root'].replace('http://','https://').rstrip('/')+'/'+relative
            path.parent.mkdir(parents=True,exist_ok=True)
            partial=path.with_name(path.name+'.part')
            try:
                with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as output:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        output.write(block)
                if sha256(partial)!=expected:raise RuntimeError('下载内容与固定版本不符: '+relative)
                partial.replace(path)
            finally:
                if partial.exists():partial.unlink()
        print('已核对 '+relative,flush=True)


if __name__=='__main__':main()
