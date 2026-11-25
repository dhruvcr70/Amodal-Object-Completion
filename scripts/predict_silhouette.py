"""Run the trained silhouette model on a same-size image/visible-mask pair."""
from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amodal.model import CompletionUNet

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    size=state['resolution']
    model=CompletionUNet(state['config']['base_channels']).to(device)
    model.load_state_dict(state['model']);model.eval()
    with Image.open(a.image) as im:image=im.convert('RGB')
    with Image.open(a.mask) as im:mask=im.convert('L')
    if image.size!=mask.size:raise ValueError('Image and mask dimensions must match.')
    visible=np.array(mask)>=128
    if not visible.any():raise ValueError('Visible mask is empty.')
    rgb=np.array(image.resize((size,size),Image.Resampling.LANCZOS),dtype='float32')/255
    vis=np.array(mask.resize((size,size),Image.Resampling.NEAREST))>=128
    x=torch.from_numpy(np.concatenate([rgb,vis[...,None].astype('float32')],2).transpose(2,0,1).copy())[None].to(device)
    with torch.inference_mode():probability=model(x).sigmoid()[0,0].cpu().numpy()
    prediction=Image.fromarray((probability>.5).astype('uint8')*255).resize(image.size,Image.Resampling.NEAREST)
    full=(np.array(prediction)>=128)|visible
    a.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(full.astype('uint8')*255).save(a.output)
    print('Saved silhouette:',a.output)
    print('This predicts a mask only; hidden RGB appearance generation is separate.')

if __name__=='__main__':main()
