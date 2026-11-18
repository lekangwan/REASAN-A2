"""汇总出生修正检查的几何、接触对象与初态一致性证据。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    if not summary['complete']:raise RuntimeError('启动检查未完整完成')
    records=[r for r in summary['spawn_records'] if r['active']]
    by_id={r['trial_id']:r for r in records}
    if len(by_id)!=summary['cases_checked']:raise RuntimeError('出生记录数量不一致')
    contact_records=json.loads((args.run/'contacts.json').read_text())
    feet=set();nonfeet=set();self_contact=set();other_contact=set()
    max_nonfoot_ground=0.;max_pair_error=0.;first_foot={}
    for record in contact_records:
        trial=record['trial_id']
        max_pair_error=max(max_pair_error,record['pair_sum_minus_net_max_n'])
        for pair in record['pairs']:
            force=float(np.linalg.norm(pair['force']))
            if pair['filter']=='/World/ground/terrain/mesh':
                if pair['sensor'].endswith('_foot'):
                    if force>1:
                        feet.add(trial);first_foot.setdefault(trial,record['time_s'])
                else:
                    max_nonfoot_ground=max(max_nonfoot_ground,force)
                    if force>1:nonfeet.add(trial)
            elif force>1:
                own=pair['sensor'].split('/Robot/')[0]
                (self_contact if pair['filter'].startswith(own+'/Robot/') else other_contact).add(trial)
    verification=json.loads((Path(__file__).resolve().parents[1]/'results/debug/20261117_230203_a2_spawn_contacts/source_sha256.json').read_text())
    root=Path(__file__).resolve().parents[1]
    for relative,digest in verification.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('原始评估数据变化: '+relative)
    result={'complete':True,'cases':len(by_id),'seconds_per_case':summary['seconds_per_case'],
        'xy_yaw_q_goal_match_source':summary['xy_yaw_q_goal_match_source'],
        'raised_cases':sum(r['height_shift_m']>1e-7 for r in by_id.values()),
        'root_height_range_m':[min(r['root_height_m'] for r in by_id.values()),max(r['root_height_m'] for r in by_id.values())],
        'minimum_actual_spawn_clearance_m':min(r['runtime_geometry_clearance_m'] for r in by_id.values()),
        'maximum_kinematics_error':max(r['kinematics_max_error'] for r in records),
        'all_spawn_clocks_zero':all(r['physics_steps']==r['inference_calls']==0 for r in records),
        'cases_with_foot_ground_force_above_1n':len(feet),
        'cases_with_nonfoot_ground_force_above_1n':sorted(nonfeet),
        'maximum_nonfoot_ground_force_n':max_nonfoot_ground,
        'cases_with_self_force_above_1n':sorted(self_contact),
        'cases_with_other_robot_force_above_1n':sorted(other_contact),
        'pair_sum_minus_net_max_n':max_pair_error,
        'first_foot_contact_time_range_s':[min(first_foot.values()),max(first_foot.values())] if first_foot else None,
        'terminations':dict(Counter(e['outcome'] for e in summary['episodes'])),
        'episodes':summary['episodes'],'control_period_checks':summary['control_period_checks'],
        'original_evaluation_files_unchanged':len(verification)}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
