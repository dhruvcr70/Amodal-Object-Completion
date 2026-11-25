"""Source-disjoint datasets and deterministic synthetic occlusion generation."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def source_split(source_id, seed=42):
    value=int(hashlib.sha256(f'{seed}:{source_id}'.encode()).hexdigest()[:12],16)/16**12
    return 'train' if value < .8 else 'val' if value < .9 else 'test'


def seed_for(key, seed=42):
    return int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8],16)


def make_shape_source(seed, size=128):
    rng=np.random.default_rng(seed)
    mask=Image.new('L',(size,size),0)
    draw=ImageDraw.Draw(mask)
    x0,y0=rng.integers(size//8,size//3,size=2)
    x1,y1=rng.integers(2*size//3,7*size//8,size=2)
    kind=int(rng.integers(0,3))
    if kind==0:
        draw.ellipse((int(x0),int(y0),int(x1),int(y1)),fill=255)
    elif kind==1:
        draw.rounded_rectangle((int(x0),int(y0),int(x1),int(y1)),radius=int(rng.integers(2,15)),fill=255)
    else:
        cx,cy=(x0+x1)/2,(y0+y1)/2
        angles=np.linspace(0,2*np.pi,int(rng.integers(4,9)),endpoint=False)+rng.uniform(0,1)
        points=[(int(cx+np.cos(a)*(x1-x0)*rng.uniform(.35,.55)),
                 int(cy+np.sin(a)*(y1-y0)*rng.uniform(.35,.55))) for a in angles]
        draw.polygon(points,fill=255)
    mask=mask.rotate(float(rng.uniform(-35,35)),resample=Image.Resampling.NEAREST)
    binary=np.asarray(mask)>0
    background=rng.uniform(.1,.9,size=3)
    foreground=(background+rng.uniform(.25,.75,size=3))%1
    image=np.broadcast_to(background,(size,size,3)).copy()
    image[binary]=foreground
    image+=rng.normal(0,.015,size=image.shape)
    return (np.clip(image,0,1)*255).astype('uint8'),binary


def occlude(image, complete_mask, seed, min_fraction=.15, max_fraction=.60):
    rng=np.random.default_rng(seed)
    complete=np.asarray(complete_mask,dtype=bool)
    if not complete.any():
        raise ValueError('A source object mask must contain foreground pixels.')
    h,w=complete.shape
    if image.shape!=(h,w,3):
        raise ValueError('Image and complete mask dimensions differ.')
    ys,xs=np.where(complete)
    for _ in range(400):
        index=int(rng.integers(len(xs)))
        cx,cy=int(xs[index]),int(ys[index])
        ow,oh=rng.integers(max(4,w//8),max(5,int(w*.8)),size=2)
        box=(int(cx-ow//2),int(cy-oh//2),int(cx+ow//2),int(cy+oh//2))
        layer=Image.new('L',(w,h),0)
        draw=ImageDraw.Draw(layer)
        if rng.random()<.5:
            draw.rectangle(box,fill=255)
        else:
            draw.ellipse(box,fill=255)
        occ=np.asarray(layer)>0
        fraction=float((complete&occ).sum()/complete.sum())
        if min_fraction<=fraction<=max_fraction:
            visible=complete&~occ
            result=image.copy()
            color=rng.integers(20,235,size=3)
            texture=rng.integers(-8,9,size=result.shape)
            result[occ]=np.clip(color+texture[occ],0,255).astype('uint8')
            return result,visible,occ,fraction
    raise ValueError('Could not generate requested occlusion; check mask size and bounds.')


def save_sample(root, source_id, variant, split, image, full, seed):
    key=f'{source_id}-{variant:02d}'
    result,visible,occluder,fraction=occlude(image,full,seed_for(key,seed))
    folder=root/split/key
    folder.mkdir(parents=True,exist_ok=False)
    items={'image':result,'visible_mask':visible.astype('uint8')*255,
           'complete_mask':full.astype('uint8')*255,'occluder_mask':occluder.astype('uint8')*255}
    record={'sample_id':key,'source_id':source_id,'split':split,'occluded_fraction':fraction}
    for name,array in items.items():
        file=folder/f'{name}.png'
        Image.fromarray(array).save(file)
        record[name]=file.relative_to(root).as_posix()
    return record


def write_manifest(root, records, origin):
    root=Path(root)
    ids={split:{r['source_id'] for r in records if r['split']==split} for split in ['train','val','test']}
    if not all(ids.values()):
        raise ValueError('All three splits need source objects. Supply more unique sources.')
    assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    summary={'origin':origin,'sources':{k:len(v) for k,v in ids.items()},
             'samples':{k:sum(r['split']==k for r in records) for k in ids},
             'occlusion_range':[min(r['occluded_fraction'] for r in records),max(r['occluded_fraction'] for r in records)]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


class CompletionDataset(Dataset):
    def __init__(self,root,split,augment_occlusion=False,augmentation_probability=.7,seed=42,
                 augmentation_min_occlusion=.08,augmentation_max_occlusion=.40,augmentation_max_extent=.50):
        self.root=Path(root)
        self.records=[json.loads(line) for line in (self.root/'manifest.jsonl').read_text().splitlines() if line.strip()]
        self.records=[r for r in self.records if r['split']==split]
        if not self.records:
            raise ValueError(f'Empty dataset split: {split}')
        self.augment_occlusion=bool(augment_occlusion and split=='train')
        self.augmentation_probability=float(augmentation_probability)
        if not 0<=self.augmentation_probability<=1:
            raise ValueError('augmentation_probability must be between 0 and 1.')
        self.augmentation_min_occlusion=float(augmentation_min_occlusion)
        self.augmentation_max_occlusion=float(augmentation_max_occlusion)
        self.augmentation_max_extent=float(augmentation_max_extent)
        if not 0<=self.augmentation_min_occlusion<self.augmentation_max_occlusion<=1:
            raise ValueError('Occlusion bounds must satisfy 0 <= minimum < maximum <= 1.')
        if not 0.1<=self.augmentation_max_extent<=1:
            raise ValueError('augmentation_max_extent must be between 0.1 and 1.')
        self.seed=int(seed)
        self.epoch=0

    def set_epoch(self,epoch):
        self.epoch=int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(self.root/r['image']) as im:
            image=np.asarray(im.convert('RGB'),dtype='float32')/255
        with Image.open(self.root/r['visible_mask']) as im:
            visible=np.asarray(im.convert('L'))>=128
        with Image.open(self.root/r['complete_mask']) as im:
            full=np.asarray(im.convert('L'))>=128
        if image.shape[:2]!=full.shape or full.shape!=visible.shape:
            raise ValueError('Dataset contains mismatched image/mask dimensions.')
        if np.any(visible&~full):
            raise ValueError('Visible mask is not a subset of the complete object.')
        if self.augment_occlusion:
            # Key each augmentation by epoch and sample so it changes each epoch,
            # remains reproducible after a resume, and never touches val/test.
            rng=np.random.default_rng(seed_for(f'{self.seed}:{self.epoch}:{r["sample_id"]}'))
            if rng.random()<.5:
                image=image[:,::-1].copy()
                visible=visible[:,::-1].copy()
                full=full[:,::-1].copy()
            if rng.random()<self.augmentation_probability:
                ys,xs=np.where(full)
                h,w=full.shape
                for _ in range(200):
                    center=int(rng.integers(len(xs)))
                    ow=int(rng.integers(max(8,w//10),max(9,int(w*self.augmentation_max_extent)+1)))
                    oh=int(rng.integers(max(8,h//10),max(9,int(h*self.augmentation_max_extent)+1)))
                    box=(int(xs[center]-ow//2),int(ys[center]-oh//2),
                         int(xs[center]+ow//2),int(ys[center]+oh//2))
                    layer=Image.new('L',(w,h),0)
                    draw=ImageDraw.Draw(layer)
                    if rng.random()<.5:
                        draw.rectangle(box,fill=255)
                    else:
                        draw.ellipse(box,fill=255)
                    synthetic=np.asarray(layer)>0
                    hidden_fraction=float((full&synthetic).sum()/full.sum())
                    if self.augmentation_min_occlusion<=hidden_fraction<=self.augmentation_max_occlusion:
                        # Reuse a random scene patch as the artificial occluder,
                        # rather than painting a constant black region.
                        sx=int(rng.integers(0,w-ow+1))
                        sy=int(rng.integers(0,h-oh+1))
                        image_uint8=np.clip(image*255,0,255).astype('uint8')
                        patch=Image.fromarray(image_uint8).crop((sx,sy,sx+ow,sy+oh)).resize((ow,oh),Image.Resampling.BILINEAR)
                        patch_canvas=Image.new('RGB',(w,h))
                        patch_canvas.paste(patch,(box[0],box[1]))
                        image_uint8[synthetic]=np.asarray(patch_canvas)[synthetic]
                        image=image_uint8.astype('float32')/255
                        visible=visible&~synthetic
                        break
        inputs=torch.from_numpy(np.concatenate([image,visible[...,None].astype('float32')],axis=2).transpose(2,0,1).copy())
        target=torch.from_numpy(full[None].astype('float32'))
        return inputs,target,r['sample_id']
