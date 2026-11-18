"""核查五段录像对应的正式结果、控制轨迹及完整视频解码。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((args.out/'provenance.json').read_text())
    source=root/provenance['source_evaluation']
    selected={x['trial_id'] for x in provenance['comparisons']}
    comparisons={}
    for name in ('episodes.jsonl','trajectory.jsonl','a2_control.jsonl'):
        def rows(path):
            with path.open() as f:
                return [r for r in map(json.loads,f) if r['trial_id'] in selected]
        original=rows(source/name);replayed=rows(args.out/'replay'/name)
        assert original==replayed,'正式记录与补录不同: '+name
        comparisons[name]={'records':len(original),'exact_match':True}
    initial={r['trial_id']:r for r in json.loads((source/'initial_states.json').read_text())}
    for r in json.loads((args.out/'replay/initial_states.json').read_text()):
        assert r==initial[r['trial_id']]
    frozen=json.loads((args.out/'source_verification.json').read_text())
    for relative,digest in frozen['files_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    (args.out/'replay_verification.json').write_text(json.dumps({
        'comparisons':comparisons,'initial_states_match':True,'source_evaluation_unchanged':True},indent=2)+'\n')
    import imageio.v2 as imageio
    import imageio_ffmpeg
    videos=json.loads((args.out/'videos.json').read_text())
    result=[]
    for video in videos:
        path=args.out/video['video']
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
        reader=imageio.get_reader(path)
        meta=reader.get_meta_data();frames=reader.count_frames()
        assert frames==video['frames'] and tuple(meta['size'])==(1280,720) and meta['fps']==25
        positions=[0,frames//4,frames//2,3*frames//4,frames-1]
        for i,frame in enumerate(positions):
            imageio.imwrite(args.out/(path.stem+f'_review_{i}.jpg'),reader.get_data(frame))
        reader.close()
        result.append({'video':path.name,'complete_decode':True,'frames':frames,
            'fps':meta['fps'],'size':meta['size'],'duration_s':frames/25,'codec':meta.get('codec')})
    (args.out/'video_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'replay':comparisons,'videos':result},indent=2),flush=True)


if __name__=='__main__':main()
