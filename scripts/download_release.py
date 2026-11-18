"""下载固定 Release 的运行资产，可选完整结果与录像，并核对仓库中的摘要。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from github_auth import token,opener

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-results',action='store_true')
    parser.add_argument('--with-videos',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'release-downloads')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'docs/RELEASE_ARTIFACTS.json').read_text())
    auth=None;http=opener()
    base='https://api.github.com/repos/'+manifest['repository']
    def request(url,accept='application/vnd.github+json'):
        headers={'Accept':accept,'User-Agent':'REASAN-A2'}
        if auth:headers['Authorization']='Bearer '+auth
        return urllib.request.Request(url,headers=headers)
    available={}
    args.out.mkdir(parents=True,exist_ok=True)
    for artifact in manifest['artifacts']:
        if artifact['kind']=='results' and not args.with_results:continue
        if artifact['kind']=='videos' and not args.with_videos:continue
        path=args.out/artifact['name']
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']:
            print('已核对 '+path.name);continue
        partial=path.with_name(path.name+'.part');h=hashlib.sha256()
        try:
            # 公开附件直接下载，避免匿名 GitHub API 的共享限额。
            url=(available[path.name]['url'] if auth else
                 'https://github.com/'+manifest['repository']+'/releases/download/'+manifest['tag']+'/'+path.name)
            try:
                response=http.open(request(url,'application/octet-stream'),timeout=120)
            except urllib.error.HTTPError as error:
                if auth or error.code not in (401,403,404):raise
                auth=token()
                with http.open(request(base+'/releases/tags/'+manifest['tag']),timeout=60) as r:release=json.load(r)
                available={a['name']:a for a in release['assets']}
                response=http.open(request(available[path.name]['url'],'application/octet-stream'),timeout=120)
            with response,partial.open('wb') as output:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    output.write(block);h.update(block)
            if h.hexdigest()!=artifact['sha256']:raise RuntimeError('附件校验失败: '+path.name)
            partial.replace(path)
        finally:
            if partial.exists():partial.unlink()
        print('已下载并核对 '+path.name,flush=True)


if __name__=='__main__':main()
