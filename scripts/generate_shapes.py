"""Create a deterministic toy dataset for engineering checks, not real-object claims."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amodal.data import make_shape_source,save_sample,seed_for,source_split,write_manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('data/shapes-v1'))
    p.add_argument('--sources',type=int,default=384)
    p.add_argument('--variants',type=int,default=2)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.sources<30 or a.variants<1 or a.size<64:
        p.error('Use at least 30 sources, one variant, and 64-pixel images.')
    if a.output.exists():
        p.error('Output exists. Choose a new output folder to preserve existing data.')
    a.output.mkdir(parents=True)
    records=[]
    for i in range(a.sources):
        sid=f'shape-{i:06d}'
        image,full=make_shape_source(seed_for(sid,a.seed),a.size)
        split=source_split(sid,a.seed)
        for variant in range(a.variants):
            records.append(save_sample(a.output,sid,variant,split,image,full,a.seed))
    summary=write_manifest(a.output,records,'procedural_shapes_engineering_only')
    summary.update(seed=a.seed,resolution=a.size,variants=a.variants)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
