"""Run one or more Pix2Gestalt RGB completion candidates in isolated Colab."""
from pathlib import Path
from contextlib import nullcontext
import argparse
import gc
import json
import os
import sys
import time

def prepare_inputs(image_path, mask_path=None):
    import numpy as np
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    if mask_path is None:
        raise ValueError('Pix2Gestalt requires a visible-object mask. For obstruction-only marking, use notebooks/02_marked_obstruction_colab.ipynb.')
    mask = Image.open(mask_path).convert('L')
    if image.size != mask.size:
        raise ValueError('Image and visible mask must have identical dimensions.')
    binary = np.array(mask) >= 128
    if not binary.any() or binary.all():
        raise ValueError('Visible mask must include target pixels and background pixels.')
    image = image.resize((256,256), Image.Resampling.LANCZOS)
    mask = Image.fromarray(binary.astype('uint8')*255).resize((256,256), Image.Resampling.NEAREST)
    return np.array(image), np.repeat(np.array(mask)[...,None],3,axis=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/content/amodal-completion'))
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True,
                   help='Binary mask of the visible target object (white=visible target).')
    p.add_argument('--scene',type=Path,
                   help='Optional untouched full scene. Provide this with --crop to enable scene composition.')
    p.add_argument('--crop',type=int,nargs=4,metavar=('X0','Y0','X1','Y1'),
                   help='Crop coordinates in --scene used to create --image.')
    p.add_argument('--occluder-mask',type=Path,
                   help='Optional obstruction mask saved with the run for full-scene composition.')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--candidates',type=int,default=1,
                   help='Number of stochastic completions to generate (1-4).')
    args=p.parse_args()
    if args.steps < 1 or args.steps > 1000:
        p.error('steps must be between 1 and 1000')
    if args.candidates < 1 or args.candidates > 4:
        p.error('candidates must be between 1 and 4')
    if bool(args.scene) != bool(args.crop):
        p.error('--scene and --crop must be supplied together')
    if args.occluder_mask and not args.scene:
        p.error('--occluder-mask requires --scene and --crop')
    args.image=args.image.resolve()
    args.mask=args.mask.resolve()
    args.checkpoint=args.checkpoint.resolve()
    args.root=args.root.resolve()
    if args.scene:
        args.scene=args.scene.resolve()
    if args.occluder_mask:
        args.occluder_mask=args.occluder_mask.resolve()
    scene_metadata=None
    if args.scene:
        from PIL import Image
        with Image.open(args.scene) as scene_image:
            scene_width,scene_height=scene_image.size
            x0,y0,x1,y1=args.crop
            if not (0 <= x0 < x1 <= scene_width and 0 <= y0 < y1 <= scene_height):
                p.error(f'--crop {args.crop} is outside scene dimensions {(scene_width,scene_height)}')
            crop_aspect=(x1-x0)/(y1-y0)
        with Image.open(args.image) as model_input:
            input_aspect=model_input.width/model_input.height
        if abs(crop_aspect-input_aspect) > 1e-3:
            p.error('The scene crop aspect ratio must match --image; padding/rotation transforms are not supported.')
        scene_metadata={'crop_xyxy':args.crop,'scene_size':[scene_width,scene_height],
                        'source_scene':str(args.scene),'input_crop_size':[x1-x0,y1-y0],
                        'model_input_resolution':[256,256]}
    image, mask = prepare_inputs(args.image,args.mask)
    upstream = args.root/'third_party/pix2gestalt/pix2gestalt'
    sys.path.insert(0,str(upstream))
    os.chdir(upstream)
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from inference import run_pix2gestalt

    if not torch.cuda.is_available():
        raise RuntimeError('Select a T4 GPU runtime before running this experiment.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    config = OmegaConf.load(upstream/'configs/sd-finetune-pix2gestalt-c_concat-256.yaml')
    config.model.params.use_ema = False
    # The published checkpoint includes training state. Map it from disk instead of
    # materializing all optimizer tensors inside the limited Colab host RAM.
    checkpoint = torch.load(str(args.checkpoint),map_location='cpu',mmap=True,weights_only=False)
    state = checkpoint['state_dict']
    with torch.device('cpu'):
        model=instantiate_from_config(config.model)
    selected={}
    ema_count=0
    for name,tensor in model.state_dict().items():
        key='model_ema.'+name[len('model.'):].replace('.','') if name.startswith('model.') else ''
        if key and key in state:
            selected[name]=state[key]
            ema_count+=1
        elif name in state:
            selected[name]=state[name]
    missing,unexpected=model.load_state_dict(selected,strict=False)
    if missing:
        raise RuntimeError('Checkpoint is missing model tensors: '+str(missing[:15]))
    if not ema_count:
        raise RuntimeError('EMA weights were not found; stop rather than silently change the baseline.')
    del selected,state,checkpoint
    gc.collect()
    model.eval().requires_grad_(False).to('cuda')
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        results=run_pix2gestalt(model,'cuda',image,mask,scale=2.0,n_samples=args.candidates,
                               ddim_steps=args.steps,precision='autocast')
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    run=args.root/'runs'/time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(run/'input.png')
    Image.fromarray(mask[:,:,0]).save(run/'visible_mask.png')
    if args.scene:
        Image.open(args.scene).convert('RGB').save(run/'scene.png')
        (run/'scene_metadata.json').write_text(json.dumps(scene_metadata,indent=2)+'\n')
    if args.occluder_mask:
        Image.open(args.occluder_mask).convert('L').save(run/'occluder_mask.png')
    # Save every sample for side-by-side human selection. Keep completion.png as
    # candidate 1 so existing tools and archives remain compatible. Pix2Gestalt
    # returns an isolated RGB crop; this runner does not composite into the scene.
    for index, result in enumerate(results, start=1):
        Image.fromarray(result).save(run/f'candidate_{index:02d}.png')
    Image.fromarray(results[0]).save(run/'completion.png')
    report={'seed':args.seed,'steps':args.steps,'guidance_scale':2.0,'candidates':len(results),
            'resolution':[256,256],'gpu':torch.cuda.get_device_name(),
            'inference_seconds':elapsed,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'ema_tensors_loaded':ema_count,'torch':torch.__version__,
            'scene_metadata':scene_metadata,
            'occluder_mask_saved':bool(args.occluder_mask),
            'status':'inference_completed','mask_export':'not implemented',
            'status':'inference_completed'}
    env=args.root/'environment_manifest.json'
    if env.exists(): report['environment']=json.loads(env.read_text())
    (run/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved run:',run)

if __name__=='__main__':
    main()
