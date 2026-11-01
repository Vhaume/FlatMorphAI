# [V5] Enrichment of the eyespot metrics (eyeun_*) with the retained detector fondu_v5_jeu3E, a copy of
# enrich_metrics_yeux_unet.py (detector rgbz_hn, fraction 0.35, threshold 0.30). Changes: checkpoint
# eye_unet_cmp_fondu_poisson_v5_jeu3E, fraction 0.30 (the detector's own), counting threshold 0.61 (chosen on
# the real validation), minimum distance between peaks 7, measurement ring radii multiplied by 0.35 / 0.30 so
# the physical extents around the eye stay the same. The metric CSVs are NOT modified in place: the folder
# metrics_regeneration/<DN> is copied (first-level CSVs only) to <DN>_v5 and the eyeun_* columns are
# recomputed there. The v5 state classifier (train_modele_production_stade_v5) reads <DN>_v5.
# Graphics card, about 15 minutes (one inference per end and per image).
# Enrichment of the eyespot metrics with the U-Net detector (neighbourhood: sclera/pupil/tissue).
# Idea: the pale area around the eye, like a sclera, contrasts more and more with the dark pupil from
# State_4 to State_6. That neighbourhood is measured at the detector's positions, consistent between
# training and inference. Seed 42. Checks the device. U-Net: Ronneberger et al., MICCAI 2015,
# DOI 10.1007/978-3-319-24574-4_28.
import os,glob,csv,numpy as np,pandas as pd,cv2,torch,torch.nn as nn,torch.nn.functional as F,sys,shutil,datetime
import segmentation_models_pytorch as smp
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device:",device)
REGEN=os.path.join("..","..","Dataset_stereotype_regeneration","dataset_training_regeneration")
IMG=os.path.join(REGEN,"images"); MK=(sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass_clahe*"))) or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass*"))) or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former*"))))[-1]
DN="dataset_dedup_State_50valid_30test"
METDIR_SRC=os.path.join(REGEN,"metrics_regeneration",DN); METDIR=os.path.join(REGEN,"metrics_regeneration",DN+"_v5")
os.makedirs(METDIR,exist_ok=True)
for _f in glob.glob(os.path.join(METDIR_SRC,"*.csv")):
    if not os.path.exists(os.path.join(METDIR,os.path.basename(_f))): shutil.copyfile(_f,os.path.join(METDIR,os.path.basename(_f)))
print("metrics copied to",METDIR,":",len(glob.glob(os.path.join(METDIR,"*.csv"))),"CSV files")
CK=os.path.join("checkpoints_regeneration","eye_unet_cmp_fondu_poisson_v5_jeu3E","eye_unet_unified_best.pth")
S=384; FRAC=0.30; PEAK_THR=0.61; MD=7; MASK_THR=128; ECH=0.35/0.30   # [V5] ring radii scaled to the 0.30 fraction
NM=[0.485,0.456,0.406,0.20]; NS=[0.229,0.224,0.225,0.15]

def worm_mask(m):
    m=(m>=MASK_THR).astype(np.uint8); n,l,st,_=cv2.connectedComponentsWithStats(m,8)
    return m if n<=1 else (l==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
def orient(bgr,mask):
    ys,xs=np.nonzero(mask); pts=np.stack([xs,ys],1).astype(np.float64); c=pts.mean(0)
    _,_,vt=np.linalg.svd(pts-c,full_matrices=False); ax=vt[0]; ang=np.degrees(np.arctan2(ax[1],ax[0]))
    H,W=mask.shape; M=cv2.getRotationMatrix2D((float(c[0]),float(c[1])),ang-90,1.0)
    rb=cv2.warpAffine(bgr,M,(W,H)); rm=cv2.warpAffine((mask*255).astype(np.uint8),M,(W,H),flags=cv2.INTER_NEAREST)>127
    return rb,rm
def tip_center(rm,top):
    ys,xs=np.nonzero(rm); y0,y1=ys.min(),ys.max(); span=y1-y0+1
    yy=y0+int(0.12*span) if top else y1-int(0.12*span); row=rm[yy]; xx=int(np.where(row)[0].mean()) if row.any() else rm.shape[1]//2
    return xx,yy,span
def crop_box(rb,rm,cx,cy,span):
    h=int(FRAC*span); H,W=rm.shape; y0=max(0,int(cy)-h); y1=min(H,int(cy)+h); x0=max(0,int(cx)-h); x1=min(W,int(cx)+h)
    g3=rb.copy(); g3[~rm]=0; c=g3[y0:y1,x0:x1]
    if c.size==0: return None
    ch,cw=c.shape[:2]; side=max(ch,cw); sq=np.zeros((side,side,3),np.uint8); oy=(side-ch)//2; ox=(side-cw)//2
    sq[oy:oy+ch,ox:ox+cw]=c; return cv2.resize(sq,(S,S))
def zloc(im):
    g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32); lm=cv2.blur(g,(21,21)); var=cv2.blur(g*g,(21,21))-lm*lm; ls=np.sqrt(np.maximum(var,0.0))+1e-3
    return np.clip(((lm-g)/ls+1.0)/6.0,0.0,1.0)
def chans(rgb):
    zc=zloc(rgb); f=rgb.astype(np.float32)/255.0; return torch.from_numpy(np.concatenate([f.transpose(2,0,1),zc[None]],0).astype(np.float32)).to(device)
class Bk(nn.Module):
    def __init__(s,net):
        super().__init__(); s.net=net; s.register_buffer("m",torch.tensor(NM).view(1,4,1,1)); s.register_buffer("s",torch.tensor(NS).view(1,4,1,1))
    def forward(s,x): return torch.sigmoid(s.net((x-s.m)/s.s))
net=smp.Unet(encoder_name="timm-efficientnet-b0",encoder_weights=None,in_channels=4,classes=1,activation=None)
mdl=Bk(net).to(device); _c=torch.load(CK,map_location=device,weights_only=False)
mdl.load_state_dict({k:v for k,v in _c["state_dict"].items() if not k.startswith("_enc.")}); mdl.eval()
assert next(mdl.parameters()).device.type=="cuda","model is not on the graphics processor"
print("v5 detector loaded | checkpoint counting threshold %.2f | fraction %.2f"%(float(_c.get("peak_thr_eval",float("nan"))),float(_c.get("frac",float("nan")))))
def peaks(hm,thr):
    t=torch.from_numpy(hm)[None,None]; mx=F.max_pool2d(t,2*MD+1,1,MD)[0,0].numpy()
    ys,xs=np.nonzero((hm==mx)&(hm>=thr)); return sorted(zip(xs.tolist(),ys.tolist(),hm[ys,xs].tolist()),key=lambda z:-z[2])
def best_crop(bgr,mimg):
    rb,rm=orient(bgr,worm_mask(mimg)); best=None
    for top in (True,False):
        cx,cy,span=tip_center(rm,top); cb=crop_box(rb,rm,cx,cy,span)
        if cb is None: continue
        rgb=cv2.cvtColor(cb,cv2.COLOR_BGR2RGB)
        with torch.no_grad(): pr=mdl(chans(rgb)[None])[0,0].cpu().numpy()
        if best is None or pr.max()>best[0]: best=(float(pr.max()),rgb,pr)
    return best
def ring(gray,cx,cy,r0,r1):
    H,W=gray.shape; yy,xx=np.ogrid[:H,:W]; d2=(yy-cy)**2+(xx-cx)**2; m=(d2>=r0*r0)&(d2<r1*r1)
    v=gray[m]; return float(np.median(v)) if v.size>=4 else np.nan
def features(bgr,mimg):
    bc=best_crop(bgr,mimg)
    if bc is None: return None
    conf,rgb,pr=bc; g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    mxc=rgb.max(2).astype(np.float32); mnc=rgb.min(2).astype(np.float32); white=mnc-(mxc-mnc)
    pk=peaks(pr,PEAK_THR); ndet=len(pk); feats=[]
    for cx,cy,v in pk[:4]:
        pup=ring(g,cx,cy,0,3*ECH); scl=ring(g,cx,cy,3*ECH,7*ECH); tis=ring(g,cx,cy,9*ECH,17*ECH); per=ring(g,cx,cy,18*ECH,30*ECH)
        wscl=ring(white,cx,cy,3*ECH,7*ECH); wtis=ring(white,cx,cy,9*ECH,17*ECH)
        feats.append((scl-pup,pup,tis-scl,per-scl,wscl-wtis))
    if feats:
        a=np.nanmean(np.array(feats),0)
        return dict(eyeun_conf=conf,eyeun_ndet=ndet,eyeun_scl_pup=a[0],eyeun_pupil=a[1],
                    eyeun_tis_scl=a[2],eyeun_grad_pig=a[3],eyeun_white_scl=a[4])
    return dict(eyeun_conf=conf,eyeun_ndet=0,eyeun_scl_pup=np.nan,eyeun_pupil=np.nan,
                eyeun_tis_scl=np.nan,eyeun_grad_pig=np.nan,eyeun_white_scl=np.nan)

# index of image and mask files by base name (all states)
IMGIDX={}; MKIDX={}
for s in range(1,7):
    for p in glob.glob(os.path.join(IMG,"State_%d"%s,"*.jpg")): IMGIDX.setdefault(os.path.basename(p),p)
    for p in glob.glob(os.path.join(MK,"State_%d"%s,"*.jpg")): MKIDX.setdefault(os.path.basename(p),p)
print("indexed images:",len(IMGIDX))
NEWCOLS=["eyeun_conf","eyeun_ndet","eyeun_scl_pup","eyeun_pupil","eyeun_tis_scl","eyeun_grad_pig","eyeun_white_scl"]
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BK=os.path.join(METDIR,"_backup_enrich_"+stamp); os.makedirs(BK,exist_ok=True)
CACHE={}
def get_feats(base):
    if base in CACHE: return CACHE[base]
    ip=IMGIDX.get(base); mp=MKIDX.get(base)
    if ip is None or mp is None: r={k:np.nan for k in NEWCOLS}
    else:
        bgr=cv2.imread(ip); mimg=cv2.imread(mp,0)
        r=features(bgr,mimg) if (bgr is not None and mimg is not None) else {k:np.nan for k in NEWCOLS}
        if r is None: r={k:np.nan for k in NEWCOLS}
    CACHE[base]=r; return r
for sp in ("train","valid","test"):
    p=os.path.join(METDIR,f"{DN}_{sp}_regeneration_metrics_eyespig.csv")
    shutil.copyfile(p,os.path.join(BK,os.path.basename(p)))
    d=pd.read_csv(p,sep=";"); n=len(d)
    add={k:[] for k in NEWCOLS}
    for i,b in enumerate(d["image"].astype(str)):
        f=get_feats(os.path.basename(b))
        for k in NEWCOLS: add[k].append(f.get(k,np.nan))
        if (i+1)%150==0: print(f"  {sp}: {i+1}/{n}")
    for k in NEWCOLS: d[k]=add[k]
    d.to_csv(p,sep=";",index=False)
    print(f"{sp}: {n} rows enriched (+{len(NEWCOLS)} columns)")
print("\nbackup before enrichment:",os.path.basename(BK))
print("columns added:",NEWCOLS)
print("Then run train_modele_production_stade_v5 (composites uni_final_v5 and metrics <DN>_v5).")