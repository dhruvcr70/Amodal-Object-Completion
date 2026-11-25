"""Train/evaluate the compact silhouette model with resumable local GPU runs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amodal.data import CompletionDataset
from amodal.model import CompletionUNet,completion_loss,sample_iou


@torch.inference_mode()
def evaluate(model,loader,device,use_amp=False,hidden_positive_weight=1.):
    model.eval()
    sums={'loss':0.,'iou':0.,'hidden_iou':0.,'visible_only_iou':0.}
    total=0
    bucket_names=['0-15%','15-30%','30-45%','45-60%','60-90%','90-100%']
    buckets={name:{'iou':[],'hidden_iou':[],'visible_only_iou':[]} for name in bucket_names}
    for x,y,_ in loader:
        x,y=x.to(device),y.to(device)
        with torch.autocast(device.type,dtype=torch.float16,enabled=use_amp):
            logits=model(x)
        visible=x[:,3:4]>.5
        truth=y>.5
        prediction=(logits.float().sigmoid()>.5)|visible
        scores=sample_iou(prediction,truth)
        hidden_scores=sample_iou(prediction&~visible,truth&~visible)
        baseline=sample_iou(visible,truth)
        n=len(x)
        sums['loss']+=float(completion_loss(logits,y,x[:,3:4]>.5,hidden_positive_weight))*n
        sums['iou']+=float(scores.sum())
        sums['hidden_iou']+=float(hidden_scores.sum())
        sums['visible_only_iou']+=float(baseline.sum())
        severity=((truth&~visible).sum((1,2,3))/truth.sum((1,2,3)).clamp_min(1)).cpu().tolist()
        for fraction,full_score,hidden_score,baseline_score in zip(
                severity,scores.cpu().tolist(),hidden_scores.cpu().tolist(),baseline.cpu().tolist()):
            bucket='0-15%' if fraction<.15 else '15-30%' if fraction<.30 else '30-45%' if fraction<.45 else '45-60%' if fraction<.60 else '60-90%' if fraction<.90 else '90-100%'
            buckets[bucket]['iou'].append(full_score)
            buckets[bucket]['hidden_iou'].append(hidden_score)
            buckets[bucket]['visible_only_iou'].append(baseline_score)
        total+=n
    return {**{k:v/total for k,v in sums.items()},'samples':total,
            'iou_by_occlusion':{k:{'samples':len(v['iou']),**{m:float(np.mean(scores)) if scores else None for m,scores in v.items()}} for k,v in buckets.items()}}


@torch.inference_mode()
def save_preview(model,dataset,device,dest,use_amp=False,is_shapes=True,split_label='test'):
    model.eval()
    n=min(6,len(dataset))
    size=dataset[0][0].shape[-1]
    gap=10
    top=48
    labels=['Occluded input','Visible mask','Full target','Predicted full','Hidden errors']
    canvas=Image.new('RGB',(size*5+gap*6,top+n*(size+gap)+gap),'#f1f5f9')
    draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    except OSError: font=ImageFont.load_default()
    caption='PROCEDURAL SHAPES: pipeline validation, not real-object performance' if is_shapes else f'{split_label.upper()} OBJECT SILHOUETTES: green correct, red extra, blue missing'
    draw.text((gap,5),caption,fill='#0f172a',font=font)
    for i,label in enumerate(labels):draw.text((gap+i*(size+gap),28),label,fill='#0f172a',font=font)
    for row in range(n):
        x,y,_=dataset[row]
        with torch.autocast(device.type,dtype=torch.float16,enabled=use_amp):
            probability=model(x[None].to(device)).float().sigmoid().cpu()[0,0].numpy()
        visible=x[3].numpy()>.5
        truth=y[0].numpy()>.5
        prediction=(probability>.5)|visible
        rgb=(x[:3].permute(1,2,0).numpy()*255).astype('uint8')
        error=np.zeros_like(rgb)+25
        error[visible]=[130,140,155]
        error[(prediction&truth)&~visible]=[35,185,105]
        error[prediction&~truth]=[240,75,75]
        error[truth&~prediction]=[65,130,240]
        panels=[rgb,visible.astype('uint8')*255,truth.astype('uint8')*255,prediction.astype('uint8')*255,error]
        for col,panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel).convert('RGB'),(gap+col*(size+gap),top+row*(size+gap)))
    canvas.save(dest)


def plot_history(history,path,is_shapes=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs=[r['epoch'] for r in history]
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].plot(epochs,[r['train_loss'] for r in history],label='Train')
    ax[0].plot(epochs,[r['val']['loss'] for r in history],label='Validation')
    ax[0].set(xlabel='Epoch',ylabel='BCE + Dice loss')
    ax[1].plot(epochs,[r['val']['iou'] for r in history],label='Completed mask')
    ax[1].plot(epochs,[r['val']['visible_only_iou'] for r in history],label='Visible-only baseline',linestyle='--')
    ax[1].set(xlabel='Epoch',ylabel='Validation mean IoU',ylim=(0,1))
    for item in ax:item.legend();item.grid(alpha=.2)
    fig.suptitle('Procedural-shape engineering experiment' if is_shapes else 'Object silhouette experiment')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/shapes-local.json'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--init-checkpoint',type=Path,
                   help='Initialize model weights from a checkpoint, then train with a fresh optimizer.')
    p.add_argument('--skip-test',action='store_true',help='Save validation-only results while tuning; do not evaluate the held-out test split.')
    a=p.parse_args()
    if a.resume and a.init_checkpoint:
        p.error('--resume and --init-checkpoint cannot be used together.')
    cfg=json.loads(a.config.read_text())
    root=Path(__file__).resolve().parents[1]
    dataset=(root/cfg['dataset']).resolve()
    run=(root/cfg['run']).resolve()
    if run.exists() and not a.resume:
        p.error('Run folder exists. Use --resume or a new run path in the config.')
    if a.resume and not (run/'last.pt').exists():p.error('No resumable last.pt checkpoint.')
    run.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256((dataset/'manifest.jsonl').read_bytes()).hexdigest()
    is_shapes=json.loads((dataset/'summary.json').read_text())['origin']=='procedural_shapes_engineering_only'
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp=device.type=='cuda' and cfg.get('precision','fp32')=='fp16'
    train=CompletionDataset(dataset,'train',
        augment_occlusion=cfg.get('augment_occlusion',False),
        augmentation_probability=cfg.get('augmentation_probability',.7),
        seed=cfg['seed'],
        augmentation_min_occlusion=cfg.get('augmentation_min_occlusion',.08),
        augmentation_max_occlusion=cfg.get('augmentation_max_occlusion',.40),
        augmentation_max_extent=cfg.get('augmentation_max_extent',.50))
    val=CompletionDataset(dataset,'val')
    model=CompletionUNet(cfg['base_channels']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    start_epoch=0;best=-1.;history=[]
    if a.init_checkpoint:
        initial=torch.load(a.init_checkpoint,map_location=device,weights_only=False)
        if initial.get('config',{}).get('base_channels')!=cfg['base_channels']:
            raise ValueError('Initial checkpoint base_channels does not match the training configuration.')
        model.load_state_dict(initial['model'])
        cfg['initialized_from']=str(a.init_checkpoint.resolve())
        del initial
    if a.resume:
        state=torch.load(run/'last.pt',map_location=device,weights_only=False)
        if state['dataset_fingerprint']!=fingerprint:raise ValueError('Dataset changed since checkpoint.')
        defaults={'augment_occlusion':False,'augmentation_probability':.7,'augmentation_min_occlusion':.08,'augmentation_max_occlusion':.40,'augmentation_max_extent':.50,'hidden_positive_weight':1.}
        for key in ['base_channels','seed','batch_size','learning_rate','precision','augment_occlusion','augmentation_probability','augmentation_min_occlusion','augmentation_max_occlusion','augmentation_max_extent','hidden_positive_weight']:
            if state['config'].get(key,defaults.get(key))!=cfg.get(key,defaults.get(key)):raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);start_epoch=state['epoch'];best=state['best_iou'];history=state['history']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    options={'batch_size':cfg['batch_size'],'num_workers':0,'pin_memory':device.type=='cuda'}
    valloader=DataLoader(val,shuffle=False,**options)
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}; train: {len(train)}; validation: {len(val)}',flush=True)
    for epoch in range(start_epoch,cfg['epochs']):
        train.set_epoch(epoch)
        model.train();running=0.;count=0;start=time.perf_counter()
        generator=torch.Generator().manual_seed(cfg['seed']+epoch)
        loader=DataLoader(train,shuffle=True,generator=generator,**options)
        for x,y,_ in loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=use_amp):logits=model(x)
            loss=completion_loss(logits,y,x[:,3:4]>.5,cfg.get('hidden_positive_weight',1.))
            if not torch.isfinite(loss):raise RuntimeError(f'Training produced non-finite loss at epoch {epoch+1}, batch {count//cfg["batch_size"]+1}. Try precision=fp32.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            running+=float(loss.detach())*len(x);count+=len(x)
        stats=evaluate(model,valloader,device,use_amp,cfg.get('hidden_positive_weight',1.))
        improved=stats['iou']>best
        best=max(best,stats['iou'])
        record={'epoch':epoch+1,'train_loss':running/count,'val':stats,'seconds':time.perf_counter()-start}
        history.append(record)
        payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                 'epoch':epoch+1,'best_iou':best,'config':cfg,'history':history,'dataset_fingerprint':fingerprint,
                 'resolution':int(train[0][0].shape[-1]),'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None}
        # Replace complete checkpoint files atomically so interruptions do not
        # destroy the previous resumable checkpoint.
        temporary=run/'last.pt.tmp'
        torch.save(payload,temporary);temporary.replace(run/'last.pt')
        if improved:
            temporary=run/'best.pt.tmp'
            torch.save(payload,temporary);temporary.replace(run/'best.pt')
        (run/'history.json').write_text(json.dumps(history,indent=2))
        print(f"Epoch {epoch+1}/{cfg['epochs']} | loss {running/count:.4f} | val IoU {stats['iou']:.4f} | visible baseline {stats['visible_only_iou']:.4f}",flush=True)
    state=torch.load(run/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model'])
    if a.skip_test:
        evaluation_data=val
        stats=evaluate(model,valloader,device,use_amp,cfg.get('hidden_positive_weight',1.))
        evaluation_split='validation_only_tuning'
    else:
        evaluation_data=CompletionDataset(dataset,'test')
        stats=evaluate(model,DataLoader(evaluation_data,shuffle=False,**options),device,use_amp,cfg.get('hidden_positive_weight',1.))
        evaluation_split='test'
    report={'scope':'procedural_shape_pipeline_check' if is_shapes else 'object_silhouette_experiment',
            'config':cfg,'dataset_fingerprint':fingerprint,'best_epoch':state['epoch'],
            'best_validation_iou':best,'evaluation_split':evaluation_split,
            'validation_tuning_metrics':stats if a.skip_test else None,
            'test':None if a.skip_test else stats,'device':str(device),
            'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
            'parameters':sum(p.numel() for p in model.parameters()),
            'execution_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
            'execution_note':'Current invocation only; a completed --resume run evaluates without further training.' if start_epoch>=cfg['epochs'] else 'Includes training and evaluation in this invocation.',
            'output_rule':'prediction union visible mask; visible pixels preserved',
            'limitations':'Synthetic shapes do not establish performance on photographs.' if is_shapes else 'Results apply to this dataset and split; hidden appearance is not reconstructed.'}
    (run/'metrics.json').write_text(json.dumps(report,indent=2))
    save_preview(model,evaluation_data,device,run/'preview.png',use_amp,is_shapes,
                 split_label='validation' if a.skip_test else 'test')
    plot_history(history,run/'training.png',is_shapes)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
