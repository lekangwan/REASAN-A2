"""发布已完成本地验证的私有交接版本；复用 Git 凭据，不保存认证信息。"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from github_auth import token,opener

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--body-file',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args()
    args.checkout=args.checkout.resolve()
    manifest=json.loads((args.checkout/'docs/RELEASE_ARTIFACTS.json').read_text())
    repository=manifest['repository'];owner,name=repository.split('/');tag=manifest['tag']
    if repository!='lekangwan/REASAN-A2':raise RuntimeError('该入口仅用于用户指定的私有交接仓库')
    if subprocess.check_output(['git','-C',str(args.checkout),'status','--porcelain'],text=True).strip():
        raise RuntimeError('发布副本有未提交改动')
    commit=subprocess.check_output(['git','-C',str(args.checkout),'rev-parse','HEAD'],text=True).strip()
    auth=token();http=opener();base='https://api.github.com'
    def api(path,data=None,method=None):
        payload=json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(base+path,data=payload,method=method,headers={
            'Authorization':'Bearer '+auth,'Accept':'application/vnd.github+json',
            'Content-Type':'application/json','User-Agent':'REASAN-handoff'})
        with http.open(req,timeout=120) as response:return json.load(response)
    if api('/user')['login']!=owner:raise RuntimeError('当前 GitHub 登录账号与目标不符')
    try:repo=api('/repos/'+repository)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        repo=api('/user/repos',{'name':name,'private':True,'auto_init':False,
            'description':'REASAN Go2 reproduction and A2 MoE zero-shot transfer; simulation, evaluation and handoff'})
    if not repo['private']:raise RuntimeError('目标仓库必须为私有')
    remote=subprocess.run(['git','-C',str(args.checkout),'remote','get-url','origin'],capture_output=True,text=True)
    if remote.returncode==0:
        if remote.stdout.strip()!=repo['clone_url']:raise RuntimeError('origin 与目标仓库不同')
    else:subprocess.run(['git','-C',str(args.checkout),'remote','add','origin',repo['clone_url']],check=True)
    subprocess.run(['git','-C',str(args.checkout),'push','-u','origin','main'],check=True)
    try:release=api('/repos/'+repository+'/releases/tags/'+tag)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        release=api('/repos/'+repository+'/releases',{'tag_name':tag,'target_commitish':commit,
            'name':tag+' — Go2 reproduction and A2 zero-shot transfer','draft':True,
            'body':args.body_file.read_text()})
    expected=manifest['artifacts']+[{'name':'SHA256SUMS',
        'sha256':hashlib.sha256((args.artifacts/'SHA256SUMS').read_bytes()).hexdigest(),
        'bytes':(args.artifacts/'SHA256SUMS').stat().st_size}]
    existing={a['name']:a for a in release['assets']}
    uploaded=[]
    for item in expected:
        path=args.artifacts/item['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('本地附件不同: '+path.name)
        if path.name in existing:
            asset=existing[path.name]
        else:
            address=release['upload_url'].split('{')[0]+'?'+urllib.parse.urlencode({'name':path.name})
            url=urllib.parse.urlsplit(address)
            connection=http.client.HTTPSConnection(url.netloc,timeout=600)
            connection.putrequest('POST',url.path+'?'+url.query)
            for key,value in {'Authorization':'Bearer '+auth,'Accept':'application/vnd.github+json',
                'User-Agent':'REASAN-handoff','Content-Type':'application/octet-stream',
                'Content-Length':str(path.stat().st_size)}.items():connection.putheader(key,value)
            connection.endheaders()
            print('UPLOADING',path.name,path.stat().st_size,flush=True)
            with path.open('rb') as f:
                while True:
                    block=f.read(1024*1024)
                    if not block:break
                    connection.send(block)
            response=connection.getresponse();body=response.read();connection.close()
            if response.status!=201:raise RuntimeError('上传失败 '+path.name+' HTTP '+str(response.status))
            asset=json.loads(body)
        if asset['size']!=item['bytes'] or asset.get('digest')!='sha256:'+item['sha256']:
            raise RuntimeError('远端附件摘要或大小不符: '+path.name)
        uploaded.append({'name':path.name,'size':asset['size'],'digest':asset['digest']})
        print('VERIFIED',path.name,flush=True)
    tag_check=subprocess.run(['git','-C',str(args.checkout),'rev-parse',tag],capture_output=True,text=True)
    if tag_check.returncode==0 and tag_check.stdout.strip()!=commit:raise RuntimeError('已有本地标签指向其他提交')
    if tag_check.returncode!=0:subprocess.run(['git','-C',str(args.checkout),'tag',tag,commit],check=True)
    subprocess.run(['git','-C',str(args.checkout),'push','origin',tag],check=True)
    release=api('/repos/'+repository+'/releases/'+str(release['id']),
        {'draft':False,'target_commitish':commit,'body':args.body_file.read_text()},method='PATCH')
    repo=api('/repos/'+repository)
    remote_refs=subprocess.check_output(['git','-C',str(args.checkout),'ls-remote','origin','refs/heads/main','refs/tags/'+tag],text=True)
    if {line.split()[0] for line in remote_refs.splitlines()}!={commit}:raise RuntimeError('远端 main 与标签未指向最终提交')
    if not repo['private'] or release['draft']:raise RuntimeError('发布状态不正确')
    if subprocess.check_output(['git','-C',str(args.checkout),'status','--porcelain'],text=True).strip():raise RuntimeError('发布后副本不干净')
    record={'repository':repo['html_url'],'private':True,'release':release['html_url'],'tag':tag,
        'commit':commit,'main_and_tag_match':True,'clean_worktree':True,'assets':uploaded}
    args.record.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
