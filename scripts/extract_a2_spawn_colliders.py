"""在固定 Isaac 容器中导出 A2 实际 USD 碰撞形状，供出生几何检查使用。"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('请使用新的文件，避免覆盖已冻结的碰撞描述')
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cpu',kit_args='--/plugins/carb.tasking.plugin/threadCount=4').app
    try:
        from a2_adapter.spawn_geometry import ROOT,extract_usd_colliders
        description=extract_usd_colliders(ROOT/'assets/a2/usd_v1/a2.usd')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(description,indent=2,allow_nan=False)+'\n')
        print('USD collision shapes:',len(description['shapes']),flush=True)
    finally:
        app.close(wait_for_replicator=False)


if __name__=='__main__':main()
