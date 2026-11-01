#!/usr/bin/env python
# coding: utf-8
"""# Eye and pupil maturation, descriptors for the State_5 against State_6 boundary

The State_5 against State_6 boundary concentrates 183 confused images, more than double the 88 of State_2
against State_3, and it is now the main source of errors. The blastema can do nothing about it: two
independent trainings showed as much, its proportion plateaus at 0.628 and it is an early-phase
marker. The best descriptor available on that boundary is eyeun_pupil, at 0.720, and it comes from the
eye detector. So that is the measure to enrich, not to replace.

What the current descriptors measure and what is missing. The existing measures take rings of FIXED
radius around the detector peak, 0 to 3 pixels for the pupil and 3 to 7 for the pale crown. A
mature eye is larger, deeper and sharper than an emerging eye, yet a fixed ring cannot see
that growth: it always measures in the same place. The seven descriptors added here measure the size
and the sharpness instead of assuming them.

pup_depth, depth of the pigmentation trough, tissue plateau minus the central minimum, over a median
radial profile instead of a fixed ring. pup_r50, radius where the profile climbs back to half its depth,
a size measure independent of any threshold. pup_area, number of pixels below the half-depth threshold
in the neighbourhood, the effective size of the pigment section. pup_slope, maximum radial gradient, the
sharpness of the edge, an emerging eye having a diffuse edge. pup_sym, bilateral symmetry, depth difference
between the two eyes relative to their mean, a mature pair being symmetric. pup_sep, interocular
distance relative to the head width. pup_hmspread, spread of the detector peak itself,
count of pixels above half the maximum, a mature eye giving a tighter peak.

No new annotation is required: everything is computed at the positions already provided by the
eye_unet_unified_384_rgbz_hn detector, with the exact geometry of enrich_metrics_yeux_unet_v3, which guarantees
coherence entre les deux familles de descripteurs.

The verification cell measures the AUC of each new descriptor on the five boundaries BEFORE any
change to the keeplist. The value to beat on State_5 against State_6 is 0.720. Nothing is added to the
keeplist automatically; the decision is taken on the figures.

To be run from the FasterViT_for_Image_Classification root, on GPU. Seed 42, semicolon
separator, tables backed up before modification. Imputation to 0 when no eye is detected,
jamais de valeur manquante.

Verified references, not clickable.
Ronneberger O., Fischer P., Brox T. (2015). U-Net: Convolutional Networks for Biomedical Image
Segmentation. arXiv:1505.04597. DOI 10.1007/978-3-319-24574-4_28.
Reddien P.W., Sanchez Alvarado A. (2004). Fundamentals of planarian regeneration. Annual Review of Cell
and Developmental Biology, 20, 725 a 757. DOI 10.1146/annurev.cellbio.20.010403.095114.
"""

# ## 1. Maturation descriptors, computed at the positions of the existing detector
import os,glob,sys,shutil,datetime,numpy as np,pandas as pd,cv2,torch,torch.nn as nn,torch.nn.functional as F
import segmentation_models_pytorch as smp
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
np.random.seed(42); torch.manual_seed(42)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch",torch.__version__,"| CUDA:",torch.cuda.is_available(),"| device:",device)
if device.type!="cuda": print("  WARNING: CPU, inference will be slow.")
REGEN=os.path.join("..","..","Dataset_stereotype_regeneration","dataset_training_regeneration")
IMG=os.path.join(REGEN,"images"); MK=(sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass_clahe*"))) or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former_*_2pass*"))) or sorted(glob.glob(os.path.join(REGEN,"masks_model_Mask2Former*"))))[-1]
DN="dataset_dedup_State_50valid_30test"; METDIR=os.path.join(REGEN,"metrics_regeneration",DN)
CK=os.path.join("checkpoints_regeneration","eye_unet_unified_384_rgbz_hn_f030_sansrenfort","eye_unet_unified_best.pth")
# geometry STRICTLY identical to enrich_metrics_yeux_unet_v3, so that the two families stay coherent
S=384; FRAC=0.30; PEAK_THR=0.30; MASK_THR=128
# Unification of the head crop fraction, 0.35 to 0.30, aligned with ancrage_cascade.py.
# At constant thumbnail size, shrinking the window increases the number of pixels per unit of
# worm length in the ratio FRAC_REF / FRAC. Every radius and distance expressed in pixels
# below was calibrated at FRAC_REF, so they are rescaled by ECH. Without that
# rescaling, the same numbers would denote different physical distances.
FRAC_REF = 0.35
ECH = FRAC_REF / FRAC                      # 1.1667 for FRAC = 0.30
MD = int(round(6 * ECH))                   # minimum distance between peaks, 6 at FRAC_REF, hence 7
NM=[0.485,0.456,0.406,0.20]; NS=[0.229,0.224,0.225,0.15]
RMAX=int(round(20*ECH))      # reach of the radial profile, 20 at FRAC_REF, beyond that we are in the tissue
R_TIS=int(round(12*ECH))     # radius from which we are on the tissue plateau
R_FOND=int(round(6*ECH))     # radius of the bottom of the pigment section
NEWCOLS=["pup_depth","pup_r50","pup_area","pup_slope","pup_sym","pup_sep","pup_hmspread"]

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
    yy=y0+int(0.12*span) if top else y1-int(0.12*span); row=rm[yy]
    xx=int(np.where(row)[0].mean()) if row.any() else rm.shape[1]//2
    return xx,yy,span
def crop_box(rb,rm,cx,cy,span):
    h=int(FRAC*span); H,W=rm.shape; y0=max(0,int(cy)-h); y1=min(H,int(cy)+h); x0=max(0,int(cx)-h); x1=min(W,int(cx)+h)
    g3=rb.copy(); g3[~rm]=0; c=g3[y0:y1,x0:x1]
    if c.size==0: return None
    ch,cw=c.shape[:2]; side=max(ch,cw); sq=np.zeros((side,side,3),np.uint8); oy=(side-ch)//2; ox=(side-cw)//2
    sq[oy:oy+ch,ox:ox+cw]=c; return cv2.resize(sq,(S,S))
def zloc(im):
    g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32); lm=cv2.blur(g,(21,21))
    var=cv2.blur(g*g,(21,21))-lm*lm; ls=np.sqrt(np.maximum(var,0.0))+1e-3
    return np.clip(((lm-g)/ls+1.0)/6.0,0.0,1.0)
def chans(rgb):
    zc=zloc(rgb); f=rgb.astype(np.float32)/255.0
    return torch.from_numpy(np.concatenate([f.transpose(2,0,1),zc[None]],0).astype(np.float32)).to(device)
class Bk(nn.Module):
    def __init__(s,net):
        super().__init__(); s.net=net
        s.register_buffer("m",torch.tensor(NM).view(1,4,1,1)); s.register_buffer("s",torch.tensor(NS).view(1,4,1,1))
    def forward(s,x): return torch.sigmoid(s.net((x-s.m)/s.s))
net=smp.Unet(encoder_name="timm-efficientnet-b0",encoder_weights=None,in_channels=4,classes=1,activation=None)
mdl=Bk(net).to(device)
_sd=torch.load(CK,map_location=device)["state_dict"]
mdl.load_state_dict({k:v for k,v in _sd.items() if not k.startswith("_enc.")}); mdl.eval()
print("detector loaded:",CK)

def peaks(hm,thr):
    t=torch.from_numpy(hm)[None,None]; mx=F.max_pool2d(t,2*MD+1,1,MD)[0,0].numpy()
    ys,xs=np.nonzero((hm==mx)&(hm>=thr))
    return sorted(zip(xs.tolist(),ys.tolist(),hm[ys,xs].tolist()),key=lambda z:-z[2])
def best_crop(bgr,mimg):
    rb,rm=orient(bgr,worm_mask(mimg)); best=None
    for top in (True,False):
        cx,cy,span=tip_center(rm,top); cb=crop_box(rb,rm,cx,cy,span)
        if cb is None: continue
        rgb=cv2.cvtColor(cb,cv2.COLOR_BGR2RGB)
        with torch.no_grad(): pr=mdl(chans(rgb)[None])[0,0].cpu().numpy()
        if best is None or pr.max()>best[0]: best=(float(pr.max()),rgb,pr)
    return best

# ---- median radial profile around a peak: measures the SIZE and the SHARPNESS instead of assuming them ----
_YY,_XX=np.mgrid[-RMAX:RMAX+1,-RMAX:RMAX+1]; _RR=np.sqrt(_YY**2+_XX**2)
def radial(g,cx,cy):
    H,W=g.shape
    y0,y1=cy-RMAX,cy+RMAX+1; x0,x1=cx-RMAX,cx+RMAX+1
    if y0<0 or x0<0 or y1>H or x1>W:                      # peak too close to the border: cropped window, padded
        pad=np.full((2*RMAX+1,2*RMAX+1),np.nan,np.float32)
        ys0,ys1=max(0,y0),min(H,y1); xs0,xs1=max(0,x0),min(W,x1)
        pad[ys0-y0:ys1-y0, xs0-x0:xs1-x0]=g[ys0:ys1,xs0:xs1]; win=pad
    else: win=g[y0:y1,x0:x1].astype(np.float32)
    prof=np.full(RMAX+1,np.nan,np.float32)
    for r in range(RMAX+1):
        sel=(_RR>=r)&(_RR<r+1)&np.isfinite(win)
        if sel.sum()>=3: prof[r]=float(np.median(win[sel]))
    return prof,win
def maturation(g,pr,cx,cy):
    prof,win=radial(g,cx,cy)
    tis=np.nanmedian(prof[R_TIS:RMAX+1])                   # tissue plateau, far from the eye
    cen=np.nanmin(prof[0:R_FOND])                          # bottom of the pigment section
    if not np.isfinite(tis) or not np.isfinite(cen): return None
    depth=float(tis-cen)
    half=tis-depth/2.0
    # half-recovery radius: eye size with no arbitrary threshold
    r50=float(RMAX)
    for r in range(RMAX+1):
        if np.isfinite(prof[r]) and prof[r]>=half: r50=float(r); break
    area=float(np.nansum((win<half)&(_RR<=RMAX)))          # effective size of the section
    d=np.diff(prof[:13]); slope=float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    # spread of the detector peak: a mature eye gives a tighter response
    H,W=pr.shape; ys0,ys1=max(0,cy-10),min(H,cy+11); xs0,xs1=max(0,cx-10),min(W,cx+11)
    sub=pr[ys0:ys1,xs0:xs1]; spread=float((sub>=0.5*max(pr[cy,cx],1e-6)).sum())
    return dict(depth=depth,r50=r50,area=area,slope=slope,spread=spread)
def feats(bgr,mimg):
    z={k:0.0 for k in NEWCOLS}
    bc=best_crop(bgr,mimg)
    if bc is None: return z
    conf,rgb,pr=bc
    g=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(0,0),1.0).astype(np.float32)
    pk=peaks(pr,0.15)[:2]
    if not pk:
        yy,xx=np.unravel_index(int(np.argmax(pr)),pr.shape); pk=[(int(xx),int(yy),float(pr.max()))]
    ms=[]
    for cx,cy,v in pk:
        m=maturation(g,pr,int(cx),int(cy))
        if m is not None: ms.append(m)
    if not ms: return z
    z["pup_depth"]=float(np.mean([m["depth"] for m in ms]))
    z["pup_r50"]=float(np.mean([m["r50"] for m in ms]))
    z["pup_area"]=float(np.mean([m["area"] for m in ms]))
    z["pup_slope"]=float(np.mean([m["slope"] for m in ms]))
    z["pup_hmspread"]=float(np.mean([m["spread"] for m in ms]))
    if len(ms)>=2:                                          # symmetry of the pair, 0 if a single eye
        a,b=ms[0]["depth"],ms[1]["depth"]
        z["pup_sym"]=float(abs(a-b)/max(1e-6,(abs(a)+abs(b))/2.0))
        z["pup_sep"]=float(np.hypot(pk[0][0]-pk[1][0],pk[0][1]-pk[1][1])/float(S))
    return z

IMGIDX={}; MKIDX={}
for s in range(1,7):
    for p in glob.glob(os.path.join(IMG,"State_%d"%s,"*.jpg")): IMGIDX.setdefault(os.path.basename(p),p)
    for p in glob.glob(os.path.join(MK,"State_%d"%s,"*.jpg")): MKIDX.setdefault(os.path.basename(p),p)
print("indexed images:",len(IMGIDX))
CACHE={}
def gf(base):
    if base in CACHE: return CACHE[base]
    ip=IMGIDX.get(base); mp=MKIDX.get(base)
    if ip is None or mp is None: r={k:0.0 for k in NEWCOLS}
    else:
        bgr=cv2.imread(ip); mm=cv2.imread(mp,0)
        r=feats(bgr,mm) if (bgr is not None and mm is not None) else {k:0.0 for k in NEWCOLS}
    CACHE[base]=r; return r
stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BK=os.path.join(METDIR,"_backup_pupille_"+stamp); os.makedirs(BK,exist_ok=True)
tabs={}
for sp in ("train","valid","test"):
    p=os.path.join(METDIR,f"{DN}_{sp}_regeneration_metrics_eyespig.csv")
    shutil.copyfile(p,os.path.join(BK,os.path.basename(p)))
    d=pd.read_csv(p,sep=";"); n=len(d)
    base=d["image"].astype(str).map(os.path.basename)
    for i,b in enumerate(base):
        if (i+1)%400==0: print("  %s %d/%d"%(sp,i+1,n))
        gf(b)
    for k in NEWCOLS: d[k]=[gf(b).get(k,0.0) for b in base]
    d.to_csv(p,sep=";",index=False); tabs[sp]=d
    print(f"{sp}: {n} rows enriched (+{len(NEWCOLS)} columns)")
print("backup:",os.path.basename(BK),"| columns:",NEWCOLS)


# ## 2. Verification BEFORE any decision: do these descriptors beat eyeun_pupil on State_5 / State_6 ?
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D=pd.concat(tabs.values(),ignore_index=True)
def num(s): return int(str(s).strip()[-1])
D["st"]=D["class"].map(num)
def auc(sub,col,haut):
    y=(sub["st"]==haut).astype(int).values; v=sub[col].replace([np.inf,-np.inf],np.nan).fillna(0.0).values
    if y.sum()<5 or (1-y).sum()<5 or np.std(v)==0: return float("nan")
    a=roc_auc_score(y,v); return max(a,1-a)
REF={"eyeun_pupil":0.720}
cols=[c for c in NEWCOLS if c in D.columns]+[c for c in ["eyeun_pupil","eyeun_scl_pup","blast_seg_prop"] if c in D.columns]
print("AUC per boundary (the last three rows are the existing references):")
print("  %-16s %8s %8s %8s %8s %8s"%("descriptor","S1/S2","S2/S3","S3/S4","S4/S5","S5/S6"))
res=[]
for c in cols:
    r=[auc(D[D["st"].isin([a,b])],c,b) for a,b in [(1,2),(2,3),(3,4),(4,5),(5,6)]]
    res.append(dict(descripteur=c,S1S2=r[0],S2S3=r[1],S3S4=r[2],S4S5=r[3],S5S6=r[4]))
    print("  %-16s %8.3f %8.3f %8.3f %8.3f %8.3f"%(c,*r))
R=pd.DataFrame(res); R.to_csv("pupille_auc_frontieres.csv",sep=";",index=False)
best=R.loc[R["S5S6"].idxmax()]
print("\nbest on State_5 / State_6: %s at %.3f  (reference eyeun_pupil %.3f)"%(best["descripteur"],best["S5S6"],REF["eyeun_pupil"]))
# combination of the new descriptors, 5-fold cross-validation grouped by worm
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
def ver(s):
    s=re.sub(r'^\d{4}-\d{2}-\d{2}-','',os.path.splitext(os.path.basename(str(s)))[0])
    return re.sub(r'-?J\d+-?','-',s).strip('-')
sub=D[D["st"].isin([5,6])].copy(); y=(sub["st"]==6).astype(int).values; g=sub["image"].map(ver).values
X=sub[[c for c in NEWCOLS if c in sub.columns]].replace([np.inf,-np.inf],np.nan).fillna(0.0).values
pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))
pp=cross_val_predict(pipe,X,y,cv=GroupKFold(5).split(X,y,groups=g),method="predict_proba")[:,1]
print("\nAUC of the 7 new descriptors COMBINED, State_5 against State_6 (CV grouped by worm): %.3f"%roc_auc_score(y,pp))
if "eyeun_pupil" in sub.columns:
    X2=np.c_[X,sub[["eyeun_pupil","eyeun_scl_pup"]].fillna(0.0).values]
    pp2=cross_val_predict(pipe,X2,y,cv=GroupKFold(5).split(X2,y,groups=g),method="predict_proba")[:,1]
    print("AUC adding eyeun_pupil and eyeun_scl_pup                                 : %.3f"%roc_auc_score(y,pp2))
print("\nDECISION. If the combination clearly exceeds 0.720, add the non-redundant")
print("descriptors (Pearson < 0.90) to the keeplist, then rerun the cross-validation. Otherwise add nothing: the")
print("tables keep the columns, they cost nothing, and the keeplist stays unchanged.")

