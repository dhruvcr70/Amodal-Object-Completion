"""Apply occlusion to real complete-object image/mask pairs listed in a JSONL file."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amodal.data import save_sample,source_split,write_manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,default=128)
    p.add_argument('--variants',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a new folder.')
    rows=[json.loads(line) for line in a.manifest.read_text().splitlines() if line.strip()]
    ids=[r['source_id'] for r in rows]
    if len(ids)!=len(set(ids)): p.error('source_id must be unique per source object.')
    a.output.mkdir(parents=True)
    records=[]
    for row in rows:
        # group_id must be shared by crops/objects originating in the same photo.
        group=row.get('group_id',row['source_id'])
        split=source_split(group,a.seed)
        image_path=(a.manifest.parent/row['image']).resolve()
        mask_path=(a.manifest.parent/row['complete_mask']).resolve()
        with Image.open(image_path) as im: image=im.convert('RGB')
        with Image.open(mask_path) as im: mask=im.convert('L')
        if image.size!=mask.size: raise ValueError('Source image and mask sizes differ.')
        image=np.asarray(image.resize((a.size,a.size),Image.Resampling.LANCZOS))
        full=np.asarray(mask.resize((a.size,a.size),Image.Resampling.NEAREST))>=128
        sid='object-'+__import__('hashlib').sha256(row['source_id'].encode()).hexdigest()[:16]
        for variant in range(a.variants):
            record=save_sample(a.output,sid,variant,split,image,full,a.seed)
            record.update(group_id=group,original_source_id=row['source_id'])
            records.append(record)
    print(json.dumps(write_manifest(a.output,records,'user_supplied_complete_object_pairs'),indent=2))

if __name__=='__main__': main()
