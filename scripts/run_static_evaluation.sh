#!/usr/bin/env bash
# 第一轮默认每个保留布局运行一次，共 24 条路线；参数 10 表示每布局重复 10 次。
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
evaluation_repeats="${1:-1}"
if [[ ! "$evaluation_repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "重复次数必须是正整数" >&2
  exit 2
fi
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

manifest=json.loads(Path('assets/OFFICIAL_MODELS.json').read_text())
root=Path('upstream/REASAN')
actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
if actual != manifest['commit']:
    raise SystemExit('上游提交与官方基线清单不一致')
for path,record in manifest['files'].items():
    if hashlib.sha256((root/path).read_bytes()).hexdigest() != record['sha256']:
        raise SystemExit('官方权重与清单不符: '+path)
PY
evaluation_out="results/evaluation/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_official_go2_static"
mkdir -p results/evaluation
mkdir "$evaluation_out"
echo "结果目录：$project_root/$evaluation_out"
echo "启动后逐条显示结果，完整终端输出保存在 console.log。"
if bash scripts/run_in_sim.sh /workspace/scripts/evaluate_static.py \
  --split test --num-envs 2 --repeats "$evaluation_repeats" --out "/workspace/$evaluation_out" \
  2>&1 | tee "$evaluation_out/console.log" | awk '/^\{"trial_id":/ {print; fflush()}'; then
  python3 - "$evaluation_out" <<'PY'
import json
from pathlib import Path
import sys
out=Path(sys.argv[1])
summary=json.loads((out/'summary.json').read_text())
if not summary['complete']:
    raise SystemExit('评估未完整结束，请检查 summary.json')
for family,name in [('sparse','稀疏'),('dense','密集'),('maze','迷宫')]:
    result=summary['per_family'][family]
    print('{}：到达 {} / {}，成功率 {:.1%}，结果分类 {}'.format(
        name,result['outcomes'].get('success',0),result['episodes'],result['success_rate'],result['outcomes']))
print('完整结果：'+str(out/'summary.json'))
PY
else
  echo "评估中断，请检查 $evaluation_out/console.log 和 summary.json" >&2
  tail -n 40 "$evaluation_out/console.log" >&2
  exit 1
fi
