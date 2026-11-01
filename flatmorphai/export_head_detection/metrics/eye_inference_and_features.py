#!/usr/bin/env python
# coding: utf-8

# # Eye-detector inference and descriptor extraction
# 
# Goal. Apply the trained U-Net detector to every image, find the eyes again, and derive per-image descriptors: number of eyes, pupil darkness, tissue darkness and pale rim of the white of the eye. These descriptors are then added to the metrics of the state model. random_state 42, separator ';'.

# In[ ]:

import os, glob, math, numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# In[ ]:

REGEN_ROOT = os.path.join("..", "..", "Dataset_stereotype_regeneration", "dataset_training_regeneration")
IMG = os.path.join(REGEN_ROOT, "images")
MK = (sorted(glob.glob(os.path.join(REGEN_ROOT,"masks_model_Mask2Former_*_2pass_clahe*"))) or sorted(glob.glob(os.path.join(REGEN_ROOT,"masks_model_Mask2Former_*_2pass*"))) or sorted(glob.glob(os.path.join(REGEN_ROOT,"masks_model_Mask2Former*"))))[-1]
CKPT = os.path.join("checkpoints_regeneration", "eye_unet", "eye_unet_best.pth")
DATASET_NAME = os.environ.get("REGEN_DATASET", "dataset_dedup_State_50valid_30test")
METRICS_ROOT = os.path.join(REGEN_ROOT, "metrics_regeneration", DATASET_NAME)
S = 256; classes = ["State_1","State_2","State_3","State_4","State_5","State_6"]
ck = torch.load(CKPT, map_location=device); PEAK_THR = ck.get("peak_thr",0.30); MINDIST = ck.get("peak_mindist",10)
print("checkpoint F1 val:", ck.get("val_f1"))

# In[ ]:

class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.net = nn.Sequential(
            nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1=DoubleConv(3,base); self.e2=DoubleConv(base,base*2); self.e3=DoubleConv(base*2,base*4); self.e4=DoubleConv(base*4,base*8)
        self.pool=nn.MaxPool2d(2); self.bott=DoubleConv(base*8,base*8)
        self.u4=nn.ConvTranspose2d(base*8,base*8,2,2); self.d4=DoubleConv(base*16,base*4)
        self.u3=nn.ConvTranspose2d(base*4,base*4,2,2); self.d3=DoubleConv(base*8,base*2)
        self.u2=nn.ConvTranspose2d(base*2,base*2,2,2); self.d2=DoubleConv(base*4,base)
        self.u1=nn.ConvTranspose2d(base,base,2,2); self.d1=DoubleConv(base*2,base); self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); b=self.bott(self.pool(e4))
        d=self.d4(torch.cat([self.u4(b),e4],1)); d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1)); d=self.d1(torch.cat([self.u1(d),e1],1))
        return torch.sigmoid(self.out(d))
model = UNet().to(device); model.load_state_dict(ck["state_dict"]); model.eval()
print("model loaded")

# In[ ]:

def worm_mask(m):
    m=(m>0).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def orient_crop(clean, mimg):
    mask=worm_mask(mimg); ys,xs=np.nonzero(mask)
    if ys.size<300: return None
    pts=np.stack([xs,ys],1).astype(np.float64); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c,full_matrices=False); ax=vt[0]; ang=np.degrees(np.arctan2(ax[1],ax[0]))
    H,W=mask.shape; M=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),ang-90,1.0)
    rimg=cv2.warpAffine(clean,M,(W,H)); rmask=cv2.warpAffine((mask*255).astype(np.uint8),M,(W,H),flags=cv2.INTER_NEAREST)>127
    ry,rx=np.nonzero(rmask)
    if ry.size<100: return None
    y0,y1,x0,x1=ry.min(),ry.max()+1,rx.min(),rx.max()+1; pad=int(0.08*(y1-y0))
    y0=max(0,y0-pad);x0=max(0,x0-pad);y1=min(H,y1+pad);x1=min(W,x1+pad)
    crop=rimg[y0:y1,x0:x1]; ch,cw=crop.shape[:2]; side=max(ch,cw)
    sq=np.zeros((side,side,3),np.uint8); oy=(side-ch)//2; ox=(side-cw)//2; sq[oy:oy+ch,ox:ox+cw]=crop
    crop256=cv2.resize(sq,(S,S))
    return rimg, crop256, (x0,y0,ox,oy,side)
def peaks(hm):
    mx=F.max_pool2d(hm,2*MINDIST+1,1,MINDIST); cand=(hm==mx)&(hm>=PEAK_THR)
    ys,xs=torch.nonzero(cand[0],as_tuple=True); vals=hm[0][ys,xs]
    order=torch.argsort(vals,descending=True)
    return [(int(xs[i]),int(ys[i])) for i in order.tolist()][:3]
def to_rframe(px,py,tr):
    x0,y0,ox,oy,side=tr; sc=side/float(S)
    return int(px*sc-ox+x0), int(py*sc-oy+y0)
def eye_feats_at(g,x,y):
    H,W=g.shape
    if not (0<=x<W and 0<=y<H): return None
    s=g[max(0,y-26):min(H,y+26),max(0,x-26):min(W,x+26)].astype(np.float32)
    yc=y-max(0,y-26); xc=x-max(0,x-26); yy,xx=np.ogrid[:s.shape[0],:s.shape[1]]; d=np.sqrt((yy-yc)**2+(xx-xc)**2)
    pup=s[d<=4]; scl=s[(d>=4)&(d<=9)]; tis=s[(d>=11)&(d<=20)]
    if pup.size<2 or scl.size<3 or tis.size<3: return None
    pv=float(np.percentile(pup,30)); sv=float(np.percentile(scl,85)); tv=float(np.median(tis))
    return [pv,sv,tv,sv-tv,sv-pv,tv-pv,float((scl>=tv+15).mean())]
FCOLS=["eye_n","eye_pupil","eye_sclera","eye_tissue","eye_sclera_tissue","eye_sclera_pupil","eye_tissue_pupil","eye_white_frac"]
def infer_image(path,mpath):
    clean=cv2.imread(path); mimg=cv2.imread(mpath,0)
    if clean is None or mimg is None: return None
    oc=orient_crop(clean,mimg)
    if oc is None: return [0.0]*8
    rimg,crop256,tr=oc; g=cv2.cvtColor(rimg,cv2.COLOR_BGR2GRAY)
    x=torch.from_numpy(cv2.cvtColor(crop256,cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)/255.0)[None].to(device)
    with torch.no_grad(): hm=model(x).cpu()
    pk=peaks(hm[0]); feats=[]
    for (px,py) in pk:
        rx,ry=to_rframe(px,py,tr); f=eye_feats_at(g,rx,ry)
        if f: feats.append(f)
    if not feats: return [0.0]*8
    m=np.mean(feats,0)
    return [float(len(feats))]+[float(v) for v in m]

# In[ ]:

# ---- inference on every image ----
rows=[]; t=0
for cls in classes:
    fs=sorted(glob.glob(os.path.join(IMG,cls,"*.jpg")))
    for f in fs:
        nm=os.path.basename(f); mp=os.path.join(MK,cls,nm)
        r=infer_image(f,mp)
        if r is None: continue
        rows.append([nm,cls]+r); t+=1
        if t%300==0: print(t,"images")
df=pd.DataFrame(rows,columns=["image","class"]+FCOLS)
outp=os.path.join(METRICS_ROOT,"eye_features_all.csv"); df.to_csv(outp,sep=";",index=False)
print("written",outp,"|",len(df),"images")
print(df.groupby("class")[["eye_n","eye_pupil","eye_sclera_tissue","eye_white_frac"]].mean().round(2).to_string())

# In[ ]:

# ---- merging the eye descriptors into the metric CSVs (columns added) ----
for sp in ["train","valid","test"]:
    mp=os.path.join(METRICS_ROOT, DATASET_NAME+"_"+sp+"_regeneration_metrics.csv")
    m=pd.read_csv(mp,sep=";"); before=m.shape[1]
    m=m.merge(df[["image"]+FCOLS],on="image",how="left")
    m[FCOLS]=m[FCOLS].fillna(0.0)
    outp=os.path.join(METRICS_ROOT, DATASET_NAME+"_"+sp+"_regeneration_metrics_eyes.csv")
    m.to_csv(outp,sep=";",index=False)
    print(sp,"->",outp,"| columns",before,"->",m.shape[1])
print("To train with the eyes, run the decimal notebook pointing at the _eyes metrics (see the instructions).")
