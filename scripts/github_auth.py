"""复用用户已有的 GitHub 认证；不写入令牌，不将认证头转发给其他域名。"""
import os
import shutil
import subprocess
import urllib.parse
import urllib.request


def token():
    value=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if value:return value
    if shutil.which('gh'):
        r=subprocess.run(['gh','auth','token'],capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip():return r.stdout.strip()
    r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        capture_output=True,text=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'},timeout=15)
    fields=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
    if r.returncode==0 and fields.get('password'):return fields['password']
    raise RuntimeError('没有可用的 GitHub 认证。可用 gh auth login，或用浏览器下载 Release 附件后执行 restore_release.py。')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc!=urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def opener():return urllib.request.build_opener(SafeRedirect())
