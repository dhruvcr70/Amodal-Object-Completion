"""Compact U-Net for RGB + visible-mask -> complete silhouette prediction."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(4,cout),nn.SiLU())


class CompletionUNet(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.enc1=DoubleConv(4,base)
        self.enc2=DoubleConv(base,base*2)
        self.enc3=DoubleConv(base*2,base*4)
        self.center=DoubleConv(base*4,base*8)
        self.dec3=DoubleConv(base*12,base*4)
        self.dec2=DoubleConv(base*6,base*2)
        self.dec1=DoubleConv(base*3,base)
        self.output=nn.Conv2d(base,1,1)

    def forward(self,x):
        e1=self.enc1(x)
        e2=self.enc2(F.max_pool2d(e1,2))
        e3=self.enc3(F.max_pool2d(e2,2))
        c=self.center(F.max_pool2d(e3,2))
        def up(t,skip): return F.interpolate(t,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        d3=self.dec3(torch.cat([up(c,e3),e3],1))
        d2=self.dec2(torch.cat([up(d3,e2),e2],1))
        return self.output(self.dec1(torch.cat([up(d2,e1),e1],1)))


def completion_loss(logits,target,visible=None,hidden_positive_weight=1.):
    logits=logits.float()
    probability=logits.sigmoid()
    dims=(1,2,3)
    if hidden_positive_weight<1:
        raise ValueError('hidden_positive_weight must be at least 1.')
    weights=torch.ones_like(target)
    if visible is not None and hidden_positive_weight!=1:
        hidden_positive=(target>.5)&~visible.bool()
        weights=weights+hidden_positive*(hidden_positive_weight-1)
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    bce=(bce*weights).sum(dims)/weights.sum(dims).clamp_min(1)
    dice=1-((2*(probability*target*weights).sum(dims)+1)/
            ((probability*weights).sum(dims)+(target*weights).sum(dims)+1)).mean()
    return bce.mean()+dice


def sample_iou(prediction,target):
    dims=(1,2,3)
    intersection=(prediction&target).sum(dims).float()
    union=(prediction|target).sum(dims).float()
    return torch.where(union>0,intersection/union.clamp_min(1),torch.ones_like(union))
