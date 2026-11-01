#!/usr/bin/env python
# coding: utf-8

# # Generating the 3-channel composite and the profile metrics
# 
# Goal. Produce for each image a 3-channel input for the image branch: channel 0 the whole worm for context, channel 1 a high-resolution zoom on the regenerating zone anchored on the detected eyes (falling back on the pale tip if there are no eyes), channel 2 that zoom enhanced by CLAHE. In grey levels the image branch wasted two identical channels; the head detail, invisible at 384 on the whole worm, now lives there. In parallel, a pigmentation profile along the median line from the head.
# 
# Sources. U-Net eyespot detector (Ronneberger et al., MICCAI 2015, arXiv 1505.04597). CLAHE (Zuiderveld, Graphics Gems IV 1994, ISBN 0123361559). random_state 42, separator ';'.

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
OUT_IMG = os.path.join(REGEN_ROOT, "processed_composite_384")
classes = ["State_1","State_2","State_3","State_4","State_5","State_6"]
for c in classes: os.makedirs(os.path.join(OUT_IMG, c), exist_ok=True)
S = 384; SD = 256
ck = torch.load(CKPT, map_location=device); PEAK_THR = ck.get("peak_thr",0.30); MINDIST = ck.get("peak_mindist",10)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
print("checkpoint F1:", ck.get("val_f1"))

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
print("detector loaded")

# In[ ]:

def worm_mask(m):
    m=(m>0).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def orient(bgr,mask):
    ys,xs=np.nonzero(mask); pts=np.stack([xs,ys],1).astype(np.float64); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c,full_matrices=False); ax=vt[0]; ang=np.degrees(np.arctan2(ax[1],ax[0]))
    H,W=mask.shape; M=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),ang-90,1.0)
    rbgr=cv2.warpAffine(bgr,M,(W,H)); rm=cv2.warpAffine((mask*255).astype(np.uint8),M,(W,H),flags=cv2.INTER_NEAREST)>127
    return rbgr,rm
def crop256(rbgr,rm):
    # identical to the eye_prep training preparation: bbox +8%, CENTRED square, colour
    ys,xs=np.nonzero(rm); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1; pad=int(0.08*(y1-y0))
    y0=max(0,y0-pad);x0=max(0,x0-pad);y1=min(rbgr.shape[0],y1+pad);x1=min(rbgr.shape[1],x1+pad)
    crop=rbgr[y0:y1,x0:x1]; ch,cw=crop.shape[:2]; side=max(ch,cw)
    sq=np.zeros((side,side,3),np.uint8); oy=(side-ch)//2; ox=(side-cw)//2; sq[oy:oy+ch,ox:ox+cw]=crop
    return cv2.resize(sq,(SD,SD)), (y0,x0,oy,ox,side)
def detect_eyes(crop_bgr):
    rgb=cv2.cvtColor(crop_bgr,cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(rgb.transpose(2,0,1).astype(np.float32)/255.0)[None].to(device)
    with torch.no_grad(): hm=model(x).cpu()[0]
    mx=F.max_pool2d(hm,2*MINDIST+1,1,MINDIST); cand=(hm==mx)&(hm>=PEAK_THR)
    ys,xs=torch.nonzero(cand[0],as_tuple=True); vals=hm[0][ys,xs]
    order=torch.argsort(vals,descending=True)
    return [(int(xs[i]),int(ys[i])) for i in order.tolist()][:2]
def eye_to_rframe(px,py,tr):
    y0,x0,oy,ox,side=tr; sc=side/float(SD); return px*sc-ox+x0, py*sc-oy+y0
def whole_channel(rg,rm):
    g=rg.copy(); g[~rm]=0; ys,xs=np.nonzero(rm); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    c=g[y0:y1,x0:x1]; s=max(c.shape); sq=np.zeros((s,s),np.uint8)
    sq[(s-c.shape[0])//2:(s-c.shape[0])//2+c.shape[0],(s-c.shape[1])//2:(s-c.shape[1])//2+c.shape[1]]=c
    return cv2.resize(sq,(S,S))
def head_tip(rg,rm):
    ys,xs=np.nonzero(rm); y0,y1=ys.min(),ys.max(); span=y1-y0+1
    inner=cv2.erode(rm.astype(np.uint8),np.ones((3,3),np.uint8),iterations=5).astype(bool)
    if inner.sum()<100: inner=rm
    def tip(top):
        sel=inner.copy()
        if top: sel[y0+int(0.15*span):]=False
        else: sel[:y1-int(0.15*span)]=False
        v=rg[sel]; return np.percentile(v,85) if v.size>10 else 0
    top=tip(True)>=tip(False)
    hy = y0+int(0.12*span) if top else y1-int(0.12*span)
    rowsel = rm[hy]; hx = int(np.where(rowsel)[0].mean()) if rowsel.any() else rm.shape[1]//2
    return hx,hy,span
def zoom_channel(rg,rm,anchor,span,frac=0.30):
    ax,ay=anchor; h=int(frac*span)
    y0=max(0,int(ay-h)); y1=min(rg.shape[0],int(ay+h)); x0=max(0,int(ax-h)); x1=min(rg.shape[1],int(ax+h))
    g=rg.copy(); g[~rm]=0; c=g[y0:y1,x0:x1]
    if c.size==0: return np.zeros((S,S),np.uint8)
    s=max(c.shape); sq=np.zeros((s,s),np.uint8); sq[(s-c.shape[0])//2:(s-c.shape[0])//2+c.shape[0],(s-c.shape[1])//2:(s-c.shape[1])//2+c.shape[1]]=c
    return cv2.resize(sq,(S,S))
PCOLS=["pigp_head_int","pigp_mid_int","pigp_head_tex","pigp_mid_tex","pigp_front","pigp_slope","pigp_neyes",
       "pigp_blast_pale","pigp_eye_interdist","pigp_eye_pupil","pigp_eye_sclera_ratio","pigp_eye_size","pigp_pig_grad"]
def head_features(rg,rm,head_top,reyes):
    ys,xs=np.nonzero(rm); y0,y1=ys.min(),ys.max(); span=y1-y0+1
    inner=cv2.erode(rm.astype(np.uint8),np.ones((3,3),np.uint8),iterations=5).astype(bool)
    if inner.sum()<100: inner=rm
    bmed=np.median(rg[inner]); yy=np.arange(rg.shape[0])
    def band(a,b):
        lo,hi=(a,b) if head_top else (1-b,1-a)
        sel=inner.copy(); m=(yy>=y0+lo*span)&(yy<y0+hi*span); sel[~m]=False
        v=rg[sel]; return (float(v.mean()),float(v.std()),v) if v.size>20 else (bmed,0.0,np.array([bmed]))
    hi_,ht,vhead=band(0.0,0.3); mi_,mt,_=band(0.3,0.6)
    rows=[band(i/10,(i+1)/10)[0] for i in range(10)]
    front=next((i for i in range(10) if rows[i]<bmed-4),10)/10.0
    slope=float(rows[0]-rows[5])
    # blastema: pale (translucent) fraction in the head 25%
    blast_pale=float((vhead>bmed+15).mean())
    # head->body pigment gradient (normalised)
    pig_grad=float((hi_-mi_))
    # eye descriptors at the detected positions
    def eyewin(x,y):
        H,W=rg.shape; s=rg[max(0,int(y)-24):min(H,int(y)+24),max(0,int(x)-24):min(W,int(x)+24)].astype(np.float32)
        yc=int(y)-max(0,int(y)-24); xc=int(x)-max(0,int(x)-24); Y,X=np.ogrid[:s.shape[0],:s.shape[1]]; d=np.sqrt((Y-yc)**2+(X-xc)**2)
        pup=s[d<=4]; scl=s[(d>=4)&(d<=9)]; tis=s[(d>=11)&(d<=20)]
        if pup.size<2 or scl.size<3 or tis.size<3: return None
        pv=float(np.percentile(pup,30)); sv=float(np.percentile(scl,85)); tv=float(np.median(tis))
        size=float((s[d<=10]<tv-12).sum())
        return pv,sv,tv,size
    ews=[eyewin(x,y) for (x,y) in reyes]; ews=[e for e in ews if e]
    if ews:
        eye_pupil=float(np.mean([e[0] for e in ews])); eye_scl=float(np.mean([e[1]-e[2] for e in ews])); eye_size=float(np.mean([e[3] for e in ews]))
    else:
        eye_pupil=bmed; eye_scl=0.0; eye_size=0.0
    if len(reyes)>=2:
        interd=float(np.hypot(reyes[0][0]-reyes[1][0],reyes[0][1]-reyes[1][1])/(span+1e-6))
    else: interd=0.0
    return [hi_,mi_,ht,mt,front,slope,float(len(reyes)),blast_pale,interd,eye_pupil,eye_scl,eye_size,pig_grad]


# In[ ]:

# ---- build the composite and the profile of every image ----
rows=[]; t=0
for cls in classes:
    for f in sorted(glob.glob(os.path.join(IMG,cls,"*.jpg"))):
        nm=os.path.basename(f); b=cv2.imread(f); m=cv2.imread(os.path.join(MK,cls,nm),0)
        if b is None or m is None: continue
        mask=worm_mask(m)
        if mask.sum()<300:
            comp=np.zeros((S,S,3),np.uint8); cv2.imwrite(os.path.join(OUT_IMG,cls,nm.replace(".jpg",".png")),comp)
            rows.append([nm,cls]+[0.0]*len(PCOLS)); continue
        rbgr,rm=orient(b,mask); rg=cv2.cvtColor(rbgr,cv2.COLOR_BGR2GRAY)
        c256,tr=crop256(rbgr,rm); eyes=detect_eyes(c256)
        reyes=[eye_to_rframe(px,py,tr) for (px,py) in eyes]
        hx,hy,span=head_tip(rg,rm)
        # anchor = eyes if detected, otherwise the pale tip; head direction for the profile
        if reyes:
            ax=np.mean([e[0] for e in reyes]); ay=np.mean([e[1] for e in reyes])
        else:
            ax,ay=hx,hy
        ys,xs=np.nonzero(rm); ymid=(ys.min()+ys.max())/2; head_top = ay<ymid
        c0=whole_channel(rg,rm)
        c1=zoom_channel(rg,rm,(ax,ay),span,0.30)          # channel 1: head zoom
        c2=_clahe.apply(zoom_channel(rg,rm,(ax,ay),span,0.15))  # channel 2: tight super-zoom on eyes+blastema, enhanced
        comp=cv2.merge([c0,c1,c2])
        cv2.imwrite(os.path.join(OUT_IMG,cls,nm.replace(".jpg",".png")),comp)
        prof=head_features(rg,rm,head_top,reyes)
        rows.append([nm,cls]+prof); t+=1
        if t%300==0: print(t,"images")
pdf=pd.DataFrame(rows,columns=["image","class"]+PCOLS)
pdf.to_csv(os.path.join(METRICS_ROOT,"pig_profile_all.csv"),sep=";",index=False)
print("composites written:",t,"| profiles:",len(pdf))
print(pdf.groupby("class")[["pigp_head_int","pigp_mid_tex","pigp_front","pigp_neyes"]].mean().round(2).to_string())

# In[ ]:

# ---- merge the profile into the eye-enriched metrics ----
for sp in ["train","valid","test"]:
    mp=os.path.join(METRICS_ROOT, DATASET_NAME+"_"+sp+"_regeneration_metrics_eyes.csv")
    if not os.path.exists(mp): print("missing",mp,"(run eye_inference_and_features)"); continue
    m=pd.read_csv(mp,sep=";"); before=m.shape[1]
    m=m.merge(pdf[["image"]+PCOLS],on="image",how="left"); m[PCOLS]=m[PCOLS].fillna(0.0)
    outp=os.path.join(METRICS_ROOT, DATASET_NAME+"_"+sp+"_regeneration_metrics_eyespig.csv")
    m.to_csv(outp,sep=";",index=False); print(sp,"->",os.path.basename(outp),"columns",before,"->",m.shape[1])
