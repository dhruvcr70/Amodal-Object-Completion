"""Exercise CUDA training with synthetic masks; this is not the project model."""
from pathlib import Path
import json
import importlib.util
import time

import torch
from torch import nn


class TrainingProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(4,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU())
        self.bottleneck=nn.Sequential(
            nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.up=nn.ConvTranspose2d(32,16,2,stride=2)
        self.decoder=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))

    def forward(self,x):
        skip=self.encoder(x)
        return self.decoder(torch.cat([skip,self.up(self.bottleneck(skip))],dim=1))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this project interpreter.')
    device=torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    print('PyTorch:',torch.__version__,'CUDA runtime:',torch.version.cuda,flush=True)

    # Torchvision is optional for the custom silhouette model. When present, also
    # verify one of its compiled GPU operators rather than checking only imports.
    torchvision_version=None
    nms_status='not installed; not required for this training check'
    if importlib.util.find_spec('torchvision') is not None:
        import torchvision
        from torchvision.ops import nms
        torchvision_version=torchvision.__version__
        boxes=torch.tensor([[0,0,10,10],[1,1,9,9]],device=device,dtype=torch.float32)
        keep=nms(boxes,torch.tensor([.9,.8],device=device),.5)
        assert keep.tolist()==[0], 'torchvision CUDA operation failed'
        nms_status='passed'

    size=256
    yy,xx=torch.meshgrid(torch.arange(size,device=device),torch.arange(size,device=device),indexing='ij')
    target=(((xx-128)**2+(yy-128)**2)<70**2).float()[None,None].repeat(2,1,1,1)
    visible=target.clone()
    visible[:,:,:,140:]=0
    rgb=torch.rand((2,3,size,size),device=device)*.1+visible*.8
    inputs=torch.cat([rgb,visible],dim=1)
    model=TrainingProbe().to(device)
    before=model.encoder[0].weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    scaler=torch.amp.GradScaler('cuda')
    losses=[]
    torch.cuda.synchronize()
    start=time.perf_counter()
    for step in range(16):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(inputs)
        logits=logits.float()
        probability=logits.sigmoid()
        dims=(1,2,3)
        dice=1-((2*(probability*target).sum(dims)+1)/(probability.sum(dims)+target.sum(dims)+1)).mean()
        loss=nn.functional.binary_cross_entropy_with_logits(logits,target)+dice
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    changed=not torch.equal(before,model.encoder[0].weight.detach())
    assert changed, 'Optimizer did not update model weights'
    numpy_bridge=torch.arange(4).numpy().tolist()
    assert numpy_bridge==[0,1,2,3]
    report={
        'status':'passed','test_type':'synthetic_environment_check_not_project_training',
        'torch':torch.__version__,'torchvision':torchvision_version,
        'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'gpu_memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
        'batch_size':2,'resolution':size,'optimizer_steps':len(losses),
        'mixed_precision':'float16','parameter_update_verified':changed,
        'torchvision_cuda_nms':nms_status,'numpy_bridge':'passed',
        'initial_loss':losses[0],'final_loss':losses[-1],'losses':losses,
        'training_seconds':seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
    }
    dest=Path(__file__).resolve().parents[1]/'runs/environment-check'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'gpu-training-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved:',dest/'gpu-training-check.json',flush=True)


if __name__=='__main__':
    main()
